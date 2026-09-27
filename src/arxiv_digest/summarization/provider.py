"""Provider abstraction and a local Ollama-compatible implementation."""

from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from httpx import TransportError
from mistralai.client import Mistral
from mistralai.client.errors import MistralError, NoResponseError


DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 300.0
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_MISTRAL_DIAGNOSTIC_HEADERS = {
    "x-ratelimit-limit": "X-RateLimit-Limit",
    "x-ratelimit-remaining": "X-RateLimit-Remaining",
    "x-ratelimit-reset": "X-RateLimit-Reset",
    "retry-after": "Retry-After",
    "x-request-id": "X-Request-ID",
    "request-id": "Request-ID",
    "x-mistral-request-id": "X-Mistral-Request-ID",
    "mistral-request-id": "Mistral-Request-ID",
    "x-correlation-id": "X-Correlation-ID",
    "correlation-id": "Correlation-ID",
    "x-trace-id": "X-Trace-ID",
    "trace-id": "Trace-ID",
    "traceparent": "traceparent",
}


def _extract_openrouter_content(response: Any) -> Any:
    """Read the assistant content from OpenAI-compatible mapping or SDK responses."""
    def field(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    choices = field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    message = field(choices[0], "message")
    return field(message, "content")


def _evidence_schema(*, qa: bool = False) -> dict[str, Any]:
    if qa:
        item = {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "minLength": 1},
            },
            "required": ["chunk_id"],
            "additionalProperties": False,
        }
    else:
        item = {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["summary", "problem", "method", "key_results", "limitations"],
                },
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
            },
            "required": ["field", "chunk_ids"],
            "additionalProperties": False,
        }
    return {"type": "array", "items": item}


def _openrouter_response_format(prompt: str) -> dict[str, Any]:
    """Request a strict schema matching the application's existing JSON contract."""
    content_fields = ("summary", "problem", "method", "key_results", "limitations")
    if "TASK: MAP" in prompt or "TASK: REDUCE" in prompt:
        properties = {
            field: {"type": "string", "minLength": 1} for field in content_fields
        }
        properties["evidence"] = _evidence_schema()
        name = "paper_digest"
        required = [*content_fields, "evidence"]
    elif "TASK: FINAL" in prompt:
        properties = {
            field: {"type": "string", "minLength": 1} for field in content_fields
        }
        properties["suggested_follow_up_questions"] = {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
        }
        properties["evidence"] = _evidence_schema()
        name = "paper_briefing"
        required = [*content_fields, "suggested_follow_up_questions", "evidence"]
    elif "CURRENT QUESTION:" in prompt and "PAPER EXCERPTS:" in prompt:
        properties = {
            "supported": {"type": "boolean"},
            "answer": {"type": "string", "minLength": 1},
            "evidence": _evidence_schema(qa=True),
        }
        name = "grounded_answer"
        required = ["supported", "answer", "evidence"]
    else:
        return {"type": "json_object"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Return a generated response as text."""


class ProviderConfigurationError(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


def _mistral_error_detail(error: Exception, api_key: str | None) -> str:
    status = getattr(error, "status_code", None)
    error_type = type(error).__name__

    if status == 401:
        category = "Authentication failed (HTTP 401)"
    elif status == 402:
        category = "Account or payment restriction (HTTP 402)"
    elif status == 429:
        category = "Rate limit exceeded (HTTP 429)"
    elif status in (400, 422):
        category = f"Invalid request or model (HTTP {status})"
    elif isinstance(error, (TransportError, NoResponseError)):
        category = f"Network/connection failure ({error_type})"
    elif status is not None:
        category = f"Mistral API error (HTTP {status})"
    elif isinstance(error, MistralError):
        category = f"Mistral API error ({error_type})"
    else:
        return f"Mistral request failed ({error_type})."

    detail = getattr(error, "body", None) or getattr(error, "message", None)
    if not detail and isinstance(error, (TransportError, NoResponseError)):
        detail = str(error)
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", errors="replace")

    parts = [category]
    if isinstance(detail, str) and detail:
        try:
            payload = json.loads(detail)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            detail = json.dumps(_redact_payload(payload, api_key), ensure_ascii=True)
        else:
            detail = _redact_text(detail, api_key)
        parts.append(f"body/message: {detail[:4000]}")

    headers = getattr(error, "headers", None)
    if headers is None:
        headers = getattr(getattr(error, "raw_response", None), "headers", None)
    if headers is not None:
        safe_headers = []
        for name, label in _MISTRAL_DIAGNOSTIC_HEADERS.items():
            value = headers.get(name)
            if value:
                safe_headers.append(f"{label}={_redact_text(str(value), api_key)}")
        if safe_headers:
            parts.append("headers: " + "; ".join(safe_headers))
    return ". ".join(parts)


def _redact_payload(value: Any, api_key: str | None) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for name, item in value.items():
            normalized = re.sub(r"[^a-z]", "", str(name).casefold())
            sensitive = any(
                marker in normalized
                for marker in ("authorization", "apikey", "token", "secret", "credential")
            )
            redacted[name] = "[REDACTED]" if sensitive else _redact_payload(item, api_key)
        return redacted
    if isinstance(value, list):
        return [_redact_payload(item, api_key) for item in value]
    if isinstance(value, str):
        return _redact_text(value, api_key)
    return value


def _redact_text(value: str, api_key: str | None) -> str:
    if api_key:
        value = value.replace(api_key, "[REDACTED]")
    value = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", value)
    return re.sub(
        r"(?i)(x-goog-api-key|api[_-]?key|authorization)(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        value,
    )


def _gemini_error_detail(error: Exception, api_key: str) -> str:
    status = getattr(error, "code", None)
    if not isinstance(status, (int, str)):
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        status_detail = f" (HTTP {status})"
    else:
        status_detail = ""

    message = getattr(error, "message", None)
    if not isinstance(message, str) or not message:
        message = str(error)
    message = _redact_text(message, api_key)
    return f"Gemini API request failed{status_detail}: {message[:1000]}"


@dataclass(slots=True)
class OllamaProvider:
    """Call a local Ollama server without an API key or extra client package."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> "OllamaProvider":
        load_dotenv()
        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ProviderConfigurationError(
                "Set LLM_MODEL to an installed local Ollama model (see .env.example)."
            )
        return cls(
            model=model,
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434").rstrip("/"),
        )

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, max_output_tokens=None, timeout_seconds=None)

    def generate_with_options(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        timeout_seconds: float | None = None,
    ) -> str:
        """Generate with a request-local output bound, used for briefing summaries."""
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return self._generate(
            prompt,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int | None,
        timeout_seconds: float | None,
    ) -> str:
        response_format = _openrouter_response_format(prompt)
        ollama_format: str | dict[str, Any] = "json"
        if response_format.get("type") == "json_schema":
            ollama_format = response_format["json_schema"]["schema"]
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": ollama_format,
        }
        if max_output_tokens is not None:
            body["options"] = {"num_predict": max_output_tokens}
        payload = json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds or self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Local LLM request failed: {exc}") from exc
        generated = body.get("response") if isinstance(body, dict) else None
        if not isinstance(generated, str):
            raise ProviderError("Local LLM returned an unexpected response")
        return generated


@dataclass(slots=True)
class MistralProvider:
    """Call Mistral chat completions and request a JSON-formatted response."""

    model: str
    client: Any
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_environment(cls) -> "MistralProvider":
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise ProviderConfigurationError(
                "Set MISTRAL_API_KEY to use the Mistral provider."
            )
        model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MISTRAL_MODEL
        try:
            client = Mistral(api_key=api_key)
        except Exception:
            raise ProviderError("Could not initialize the Mistral client.") from None
        return cls(model=model, client=client, api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise ProviderError(_mistral_error_detail(exc, self.api_key)) from None

        try:
            generated = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            raise ProviderError("Mistral returned an unexpected chat completion response.") from None
        if not isinstance(generated, str) or not generated.strip():
            raise ProviderError("Mistral returned an empty or non-text completion.")
        return generated


@dataclass(slots=True)
class GeminiProvider:
    """Generate grounded-LLM text through Google's Gemini API."""

    model: str
    client: Any
    api_key: str = field(repr=False)

    @classmethod
    def from_environment(cls) -> "GeminiProvider":
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ProviderConfigurationError(
                "Set GEMINI_API_KEY to use the Gemini provider."
            )
        model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
        try:
            client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise ProviderError(_gemini_error_detail(exc, api_key)) from None
        return cls(model=model, client=client, api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            generated = response.text
        except Exception as exc:
            raise ProviderError(_gemini_error_detail(exc, self.api_key)) from None
        if not isinstance(generated, str) or not generated.strip():
            raise ProviderError("Gemini returned an empty or non-text completion.")
        return generated


@dataclass(slots=True)
class OpenRouterProvider:
    """Use OpenRouter's OpenAI-compatible chat-completions endpoint."""

    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 120.0

    @classmethod
    def from_environment(cls) -> "OpenRouterProvider":
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ProviderConfigurationError(
                "Set OPENROUTER_API_KEY to use the OpenRouter provider."
            )
        model = os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_OPENROUTER_MODEL
        return cls(model=model, api_key=api_key)

    def generate(self, prompt: str) -> str:
        response_format = _openrouter_response_format(prompt)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": response_format,
                **(
                    {"provider": {"require_parameters": True}}
                    if response_format["type"] == "json_schema"
                    else {}
                ),
            }
        ).encode("utf-8")
        request = Request(
            OPENROUTER_BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                raw_body = exc.read().decode("utf-8", errors="replace")
                parsed_body = json.loads(raw_body)
            except (OSError, json.JSONDecodeError):
                parsed_body = raw_body if "raw_body" in locals() else ""
            if isinstance(parsed_body, (dict, list)):
                detail = json.dumps(_redact_payload(parsed_body, self.api_key), ensure_ascii=True)
            else:
                detail = _redact_text(str(parsed_body), self.api_key)
            message = f"OpenRouter API request failed (HTTP {exc.code})"
            if detail:
                message += f": {detail[:2000]}"
            raise ProviderError(message) from None
        except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"OpenRouter request failed ({type(exc).__name__})."
            ) from None

        generated = _extract_openrouter_content(body)
        if generated is None:
            raise ProviderError("OpenRouter returned an unexpected chat completion response.") from None
        if not isinstance(generated, str) or not generated.strip():
            raise ProviderError("OpenRouter returned an empty or non-text completion.")
        return generated


def create_llm_provider(provider: str | None = None) -> LLMProvider:
    """Create the configured provider; Ollama remains the compatibility default."""
    load_dotenv()
    selected = (provider or os.getenv("LLM_PROVIDER", "")).strip().casefold() or "ollama"
    if selected == "mistral":
        return MistralProvider.from_environment()
    if selected == "gemini":
        return GeminiProvider.from_environment()
    if selected == "openrouter":
        return OpenRouterProvider.from_environment()
    if selected == "ollama":
        return OllamaProvider.from_environment()
    raise ProviderConfigurationError(
        f"Unknown LLM_PROVIDER {selected!r}; supported values are 'ollama', 'mistral', 'gemini', and 'openrouter'."
    )
