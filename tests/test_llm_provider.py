from os import environ
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

import httpx
from mistralai.client.errors import SDKError

from arxiv_digest.summarization.provider import (
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MISTRAL_MODEL,
    GeminiProvider,
    MistralProvider,
    OllamaProvider,
    OpenRouterProvider,
    ProviderConfigurationError,
    ProviderError,
    _extract_openrouter_content,
    create_llm_provider,
)


class TestOllamaProvider(unittest.TestCase):
    def test_posts_non_streaming_json_to_standard_api(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": "LOCAL_LLM_OK"}
        ).encode("utf-8")
        provider = OllamaProvider(
            model="llama3.2:3b", base_url="http://127.0.0.1:11434"
        )

        with patch(
            "arxiv_digest.summarization.provider.urlopen", return_value=response
        ) as urlopen:
            generated = provider.generate("LOCAL_LLM_OK")

        self.assertEqual(generated, "LOCAL_LLM_OK")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "llama3.2:3b",
                "prompt": "LOCAL_LLM_OK",
                "stream": False,
                "format": "json",
            },
        )
        self.assertEqual(
            urlopen.call_args.kwargs["timeout"], DEFAULT_OLLAMA_TIMEOUT_SECONDS
        )

    def test_request_local_output_limit_uses_num_predict(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": '{"summary":"short"}'}
        ).encode("utf-8")
        provider = OllamaProvider(model="llama3.2:3b")
        with patch(
            "arxiv_digest.summarization.provider.urlopen", return_value=response
        ) as urlopen:
            provider.generate_with_options(
                'Return JSON: {"summary":"..."}',
                max_output_tokens=300,
                timeout_seconds=45,
            )
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["options"], {"num_predict": 300})
        self.assertEqual(payload["stream"], False)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 45)

    def test_structured_prompts_use_existing_json_schema(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": '{"summary":"x"}'}
        ).encode("utf-8")
        provider = OllamaProvider(model="llama3.2:3b")
        prompt = (
            "TASK: FINAL\nReturn JSON with summary, problem, method, key_results, "
            "limitations, suggested_follow_up_questions, and evidence."
        )

        with patch(
            "arxiv_digest.summarization.provider.urlopen", return_value=response
        ) as urlopen:
            provider.generate(prompt)

        payload = json.loads(urlopen.call_args.args[0].data)
        schema = payload["format"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            schema["required"],
            [
                "summary",
                "problem",
                "method",
                "key_results",
                "limitations",
                "suggested_follow_up_questions",
                "evidence",
            ],
        )
        self.assertFalse(schema["additionalProperties"])

    def test_qa_schema_requests_chunk_ids_only(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"response": '{"supported":false,"answer":"Not found","evidence":[] }'}
        ).encode("utf-8")
        prompt = "CURRENT QUESTION:\nWhat is supported?\nPAPER EXCERPTS:\n[]"
        with patch(
            "arxiv_digest.summarization.provider.urlopen", return_value=response
        ) as urlopen:
            OllamaProvider(model="llama3.2:3b").generate(prompt)

        schema = json.loads(urlopen.call_args.args[0].data)["format"]
        evidence_item = schema["properties"]["evidence"]["items"]
        self.assertEqual(evidence_item["required"], ["chunk_id"])
        self.assertEqual(set(evidence_item["properties"]), {"chunk_id"})


class TestMistralProvider(unittest.TestCase):
    def test_success_returns_only_generated_text_and_uses_json_mode(self) -> None:
        completion = Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"result": "ok"}'))]
            )
        )
        provider = MistralProvider(
            model="mistral-small-latest",
            client=SimpleNamespace(chat=SimpleNamespace(complete=completion)),
        )

        output = provider.generate("Return a JSON object.")

        self.assertEqual(output, '{"result": "ok"}')
        completion.assert_called_once_with(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Return a JSON object."}],
            response_format={"type": "json_object"},
        )

    def test_missing_api_key_has_clear_configuration_error(self) -> None:
        with patch("arxiv_digest.summarization.provider.load_dotenv"):
            with patch.dict(environ, {"MISTRAL_API_KEY": "", "LLM_MODEL": ""}):
                with self.assertRaisesRegex(ProviderConfigurationError, "MISTRAL_API_KEY"):
                    MistralProvider.from_environment()

    def test_environment_model_default_and_sdk_client_creation(self) -> None:
        fake_client = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.Mistral", return_value=fake_client) as sdk,
            patch.dict(environ, {"MISTRAL_API_KEY": "test-key", "LLM_MODEL": ""}),
        ):
            provider = MistralProvider.from_environment()

        self.assertEqual(provider.model, DEFAULT_MISTRAL_MODEL)
        self.assertIs(provider.client, fake_client)
        sdk.assert_called_once_with(api_key="test-key")

    def test_malformed_response_is_provider_error(self) -> None:
        provider = MistralProvider(
            model="mistral-small-latest",
            client=SimpleNamespace(chat=SimpleNamespace(complete=Mock(return_value={"choices": []}))),
        )
        with self.assertRaisesRegex(ProviderError, "unexpected chat completion"):
            provider.generate("prompt")

    def test_api_error_does_not_expose_api_key_or_sdk_message(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(complete=Mock(side_effect=RuntimeError("token secret-key leaked")))
        )
        provider = MistralProvider(model="mistral-small-latest", client=client)
        with self.assertRaises(ProviderError) as raised:
            provider.generate("prompt")
        self.assertNotIn("secret-key", str(raised.exception))
        self.assertNotIn("token", str(raised.exception))

    def test_sdk_status_diagnostics_are_specific_and_redact_credentials(self) -> None:
        api_key = "test-key-never-display"
        statuses = {
            401: "Authentication failed (HTTP 401)",
            402: "Account or payment restriction (HTTP 402)",
            429: "Rate limit exceeded (HTTP 429)",
            400: "Invalid request or model (HTTP 400)",
            422: "Invalid request or model (HTTP 422)",
            503: "Mistral API error (HTTP 503)",
        }
        for status, expected in statuses.items():
            with self.subTest(status=status):
                response = httpx.Response(
                    status,
                    request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
                    headers={
                        "X-RateLimit-Limit": "10",
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": "30",
                        "Retry-After": "5",
                        "X-Request-ID": "request-123",
                        "Authorization": "Bearer header-secret",
                    },
                    json={"message": f"response mentions {api_key}", "api_key": "body-secret"},
                )
                sdk_error = SDKError("API error occurred", response, response.text)
                provider = MistralProvider(
                    model="mistral-small-latest",
                    client=SimpleNamespace(chat=SimpleNamespace(complete=Mock(side_effect=sdk_error))),
                    api_key=api_key,
                )
                with self.assertRaises(ProviderError) as raised:
                    provider.generate("prompt")
                self.assertIn(expected, str(raised.exception))
                self.assertIn("[REDACTED]", str(raised.exception))
                self.assertIn("X-RateLimit-Limit=10", str(raised.exception))
                self.assertIn("X-RateLimit-Remaining=0", str(raised.exception))
                self.assertIn("X-RateLimit-Reset=30", str(raised.exception))
                self.assertIn("Retry-After=5", str(raised.exception))
                self.assertIn("X-Request-ID=request-123", str(raised.exception))
                self.assertNotIn(api_key, str(raised.exception))
                self.assertNotIn("header-secret", str(raised.exception))
                self.assertNotIn("body-secret", str(raised.exception))

    def test_network_diagnostic_identifies_connection_failure(self) -> None:
        provider = MistralProvider(
            model="mistral-small-latest",
            client=SimpleNamespace(
                chat=SimpleNamespace(complete=Mock(side_effect=httpx.ConnectError("DNS resolution failed")))
            ),
        )
        with self.assertRaisesRegex(ProviderError, "Network/connection failure.*DNS resolution failed"):
            provider.generate("prompt")


class TestGeminiProvider(unittest.TestCase):
    def test_success_returns_generated_text_with_json_mode(self) -> None:
        generate_content = Mock(return_value=SimpleNamespace(text='{"result": "ok"}'))
        provider = GeminiProvider(
            model="gemini-3.8-flash",
            client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
            api_key="test-key",
        )

        self.assertEqual(provider.generate("Return a JSON object."), '{"result": "ok"}')
        args = generate_content.call_args.kwargs
        self.assertEqual(args["model"], "gemini-3.8-flash")
        self.assertEqual(args["contents"], "Return a JSON object.")
        self.assertEqual(args["config"].response_mime_type, "application/json")

    def test_missing_api_key_has_clear_configuration_error(self) -> None:
        with patch("arxiv_digest.summarization.provider.load_dotenv"):
            with patch.dict(environ, {"GEMINI_API_KEY": "", "LLM_MODEL": ""}):
                with self.assertRaisesRegex(ProviderConfigurationError, "GEMINI_API_KEY"):
                    GeminiProvider.from_environment()

    def test_malformed_response_is_provider_error(self) -> None:
        provider = GeminiProvider(
            model="gemini-3.8-flash",
            client=SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text=None)))),
            api_key="test-key",
        )
        with self.assertRaisesRegex(ProviderError, "empty or non-text"):
            provider.generate("prompt")

    def test_api_error_redacts_key(self) -> None:
        api_key = "gemini-secret-test-key"

        class FakeAPIError(Exception):
            code = 429
            message = f"Rate limited; api_key={api_key}"

        provider = GeminiProvider(
            model=DEFAULT_GEMINI_MODEL,
            client=SimpleNamespace(
                models=SimpleNamespace(generate_content=Mock(side_effect=FakeAPIError()))
            ),
            api_key=api_key,
        )
        with self.assertRaises(ProviderError) as raised:
            provider.generate("prompt")
        self.assertIn("HTTP 429", str(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))


class TestOpenRouterProvider(unittest.TestCase):
    def test_extracts_content_from_openai_compatible_sdk_response(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Reply: OPENROUTER_OK")
                )
            ]
        )

        self.assertEqual(
            _extract_openrouter_content(response), "Reply: OPENROUTER_OK"
        )

    def test_environment_configuration_and_default_free_model(self) -> None:
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch.dict(environ, {"OPENROUTER_API_KEY": "test-openrouter-key", "OPENROUTER_MODEL": ""}),
        ):
            provider = OpenRouterProvider.from_environment()
        self.assertEqual(provider.model, "openrouter/free")
        self.assertEqual(provider.api_key, "test-openrouter-key")
        self.assertNotIn("test-openrouter-key", repr(provider))

    def test_missing_api_key_is_configuration_error(self) -> None:
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch.dict(environ, {"OPENROUTER_API_KEY": ""}),
        ):
            with self.assertRaisesRegex(ProviderConfigurationError, "OPENROUTER_API_KEY"):
                OpenRouterProvider.from_environment()

    def test_success_parses_openai_compatible_response_and_json_mode(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"summary": "grounded"}'}}]}
        ).encode("utf-8")
        provider = OpenRouterProvider(model="openrouter/free", api_key="test-key")
        with patch("arxiv_digest.summarization.provider.urlopen", return_value=response) as urlopen:
            generated = provider.generate("Return a JSON summary.")

        self.assertEqual(generated, '{"summary": "grounded"}')
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "openrouter/free")
        self.assertEqual(body["messages"], [{"role": "user", "content": "Return a JSON summary."}])
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_http_error_is_reported_without_api_key(self) -> None:
        api_key = "openrouter-secret-key"
        error_body = json.dumps(
            {"error": {"message": f"Invalid credential {api_key}", "api_key": api_key}}
        ).encode("utf-8")
        http_error = HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            BytesIO(error_body),
        )
        provider = OpenRouterProvider(model="openrouter/free", api_key=api_key)
        with patch("arxiv_digest.summarization.provider.urlopen", side_effect=http_error):
            with self.assertRaises(ProviderError) as raised:
                provider.generate("prompt")
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))

    def test_malformed_success_response_is_provider_error(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices": []}'
        provider = OpenRouterProvider(model="openrouter/free", api_key="test-key")
        with patch("arxiv_digest.summarization.provider.urlopen", return_value=response):
            with self.assertRaisesRegex(ProviderError, "unexpected chat completion"):
                provider.generate("prompt")

    def test_environment_uses_configured_model_and_sdk_client(self) -> None:
        fake_client = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.genai.Client", return_value=fake_client) as sdk,
            patch.dict(environ, {"GEMINI_API_KEY": "test-key", "LLM_MODEL": "gemini-3.8-flash"}),
        ):
            provider = GeminiProvider.from_environment()
        self.assertEqual(provider.model, "gemini-3.8-flash")
        self.assertIs(provider.client, fake_client)
        sdk.assert_called_once_with(api_key="test-key")


class TestProviderFactory(unittest.TestCase):
    def test_openrouter_selection(self) -> None:
        expected = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.OpenRouterProvider.from_environment", return_value=expected) as create,
            patch.dict(environ, {"LLM_PROVIDER": "openrouter"}),
        ):
            self.assertIs(create_llm_provider(), expected)
        create.assert_called_once_with()

    def test_gemini_selection(self) -> None:
        expected = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.GeminiProvider.from_environment", return_value=expected) as create,
            patch.dict(environ, {"LLM_PROVIDER": "gemini"}),
        ):
            self.assertIs(create_llm_provider(), expected)
        create.assert_called_once_with()

    def test_mistral_selection(self) -> None:
        expected = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.MistralProvider.from_environment", return_value=expected) as create,
            patch.dict(environ, {"LLM_PROVIDER": "mistral"}),
        ):
            self.assertIs(create_llm_provider(), expected)
        create.assert_called_once_with()

    def test_ollama_selection(self) -> None:
        expected = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.OllamaProvider.from_environment", return_value=expected) as create,
            patch.dict(environ, {"LLM_PROVIDER": "ollama"}),
        ):
            self.assertIs(create_llm_provider(), expected)
        create.assert_called_once_with()

    def test_missing_provider_defaults_to_ollama(self) -> None:
        expected = object()
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch("arxiv_digest.summarization.provider.OllamaProvider.from_environment", return_value=expected) as create,
            patch.dict(environ, {"LLM_PROVIDER": ""}),
        ):
            self.assertIs(create_llm_provider(), expected)
        create.assert_called_once_with()

    def test_unknown_provider_has_clear_configuration_error(self) -> None:
        with (
            patch("arxiv_digest.summarization.provider.load_dotenv"),
            patch.dict(environ, {"LLM_PROVIDER": "unknown"}),
        ):
            with self.assertRaisesRegex(ProviderConfigurationError, "supported values"):
                create_llm_provider()


if __name__ == "__main__":
    unittest.main()
