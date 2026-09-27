"""Question answering over chunks from one selected arXiv paper."""

import json
import os
import re
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv

from arxiv_digest.indexing.embeddings import Embedder
from arxiv_digest.qa.models import (
    NOT_FOUND_ANSWER,
    ConversationTurn,
    QAEvidence,
    QAResult,
)
from arxiv_digest.summarization.provider import LLMProvider, create_llm_provider
from arxiv_digest.vector_store.store import VectorSearchResult, VectorStore

DEFAULT_MAX_COSINE_DISTANCE = 0.81
_QUESTION_STOP_WORDS = {
    "what", "which", "who", "when", "where", "why", "how", "does", "do",
    "did", "is", "are", "was", "were", "the", "a", "an", "this", "that",
    "these", "those", "they", "them", "their", "its", "it", "paper", "author",
    "authors", "identify", "describe", "report", "state", "explain", "tell",
}


class MalformedQAResponse(ValueError):
    """Raised when the model response is invalid or cites unsupported evidence."""


class GroundedQA:
    """Embed a question, retrieve only the selected paper, and validate citations."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        provider: LLMProvider | None = None,
        top_k: int = 5,
        max_cosine_distance: float | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_cosine_distance is None:
            load_dotenv()
            try:
                max_cosine_distance = float(
                    os.getenv("QA_MAX_COSINE_DISTANCE", str(DEFAULT_MAX_COSINE_DISTANCE))
                )
            except ValueError as exc:
                raise ValueError("QA_MAX_COSINE_DISTANCE must be a number") from exc
        if not 0.0 <= max_cosine_distance <= 2.0:
            raise ValueError("max_cosine_distance must be between 0 and 2")
        self._embedder = embedder
        self._vector_store = vector_store
        self._provider = provider
        self.top_k = top_k
        self.max_cosine_distance = max_cosine_distance

    def answer(
        self,
        question: str,
        paper_id: str,
        history: Sequence[ConversationTurn] = (),
        paper_context: str = "",
    ) -> QAResult:
        if not question.strip():
            raise ValueError("question cannot be empty")
        if not paper_id.strip():
            raise ValueError("paper_id cannot be empty")
        if any(not isinstance(turn, ConversationTurn) for turn in history):
            raise TypeError("history must contain ConversationTurn values")

        retrieval_query = self._retrieval_query(question, history)
        queries = [retrieval_query]
        if paper_context.strip():
            queries.append(self._paper_context_query(paper_context, retrieval_query))
        vectors = self._embedder.embed_texts(queries)
        if len(vectors) != len(queries) or any(not vector for vector in vectors):
            raise RuntimeError("embedder returned an unexpected query vector")
        result_sets = [
            self._vector_store.search(vector, top_k=self.top_k, paper_id=paper_id)
            for vector in vectors
        ]
        direct_results = [
            result for result in result_sets[0] if result.chunk.paper_id == paper_id
        ]
        if not direct_results or min(item.distance for item in direct_results) > self.max_cosine_distance:
            return self._not_found(question, direct_results)
        selected = tuple(self._merge_results(result_sets, paper_id, self.top_k))
        if not selected:
            return self._not_found(question)
        selected_ids = {item.chunk.chunk_id for item in selected}
        selected_direct = [
            result for result in direct_results if result.chunk.chunk_id in selected_ids
        ]

        provider = self._provider or create_llm_provider()
        response = provider.generate(self._prompt(question, selected, history))
        supported, answer, chunk_ids = self._parse_response(response)
        if not supported:
            extractive = self._extractive_fallback(selected_direct)
            if extractive is not None:
                answer, evidence = extractive
                self._validate_evidence(evidence, selected)
                pages = tuple(dict.fromkeys(item.page_number for item in evidence))
                return QAResult(
                    question=question,
                    answer=answer,
                    source_chunks=tuple(item.chunk for item in selected),
                    page_numbers=pages,
                    retrieval_distances=tuple(item.distance for item in selected),
                    evidence=evidence,
                    grounded=True,
                )
            return self._not_found(question, selected)
        chunks_by_id = {result.chunk.chunk_id: result.chunk for result in selected}
        evidence = self._authoritative_evidence(chunk_ids, chunks_by_id)
        if not evidence:
            return self._not_found(question, selected)
        if not self._answer_is_source_text(answer, evidence) or not self._answer_addresses_question(
            answer, question
        ):
            extractive = self._extractive_fallback(selected_direct)
            if extractive is None:
                return self._not_found(question, selected)
            answer, evidence = extractive
        self._validate_evidence(evidence, selected)
        pages = tuple(dict.fromkeys(item.page_number for item in evidence))
        return QAResult(
            question=question,
            answer=answer,
            source_chunks=tuple(item.chunk for item in selected),
            page_numbers=pages,
            retrieval_distances=tuple(item.distance for item in selected),
            evidence=evidence,
            grounded=True,
        )

    @staticmethod
    def _paper_context_query(paper_context: str, retrieval_query: str) -> str:
        return f"Selected paper title and abstract:\n{paper_context[:1800]}\n{retrieval_query}"

    @staticmethod
    def _merge_results(
        result_sets: Sequence[Sequence[VectorSearchResult]],
        paper_id: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Fuse direct-question and paper-context rankings without mixing papers."""
        rank_constant = 60
        scores: dict[str, float] = {}
        best: dict[str, VectorSearchResult] = {}
        context_limit = max(1, top_k // 4) if top_k > 1 else 0
        direct_limit = top_k - context_limit
        for query_index, results in enumerate(result_sets):
            result_limit = direct_limit if query_index == 0 else context_limit
            if result_limit == 0:
                continue
            query_weight = 2.0 if query_index == 0 else 1.0
            paper_results = [
                result for result in results if result.chunk.paper_id == paper_id
            ][:result_limit]
            for rank, result in enumerate(paper_results, start=1):
                chunk = result.chunk
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                    query_weight / (rank_constant + rank)
                )
                previous = best.get(chunk.chunk_id)
                if previous is None or result.distance < previous.distance:
                    best[chunk.chunk_id] = result
        ordered_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], best[chunk_id].distance, chunk_id),
        )
        return [best[chunk_id] for chunk_id in ordered_ids[:top_k]]

    @staticmethod
    def _retrieval_query(
        question: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        context: list[str] = []
        if GroundedQA._is_contextual_followup(question) and history:
            previous_user = next(
                (turn.content for turn in reversed(history) if turn.role == "user"), None
            )
            previous_answer = next(
                (turn.content for turn in reversed(history) if turn.role == "assistant"), None
            )
            if previous_user:
                context.append(f"Previous question: {previous_user}")
            if previous_answer:
                context.append(f"Previous answer for reference: {previous_answer}")
        return "\n".join([*context, f"Current question: {question}"])

    @staticmethod
    def _is_contextual_followup(question: str) -> bool:
        return bool(
            re.search(
            r"\b(it|they|them|its|their|those|these|that|one|above)\b|\bhow\s+many\b|\bhow\s+large\b|\bwhat\s+about\b",
            question.casefold(),
            )
        )

    @staticmethod
    def _prompt(
        question: str,
        results: Sequence[VectorSearchResult],
        history: Sequence[ConversationTurn],
    ) -> str:
        excerpts = [
            {
                "chunk_id": item.chunk.chunk_id,
                "paper_id": item.chunk.paper_id,
                "page_numbers": list(item.chunk.page_numbers),
                "section": item.chunk.section,
                "text": item.chunk.text,
            }
            for item in results
        ]
        turns = [
            {"role": turn.role, "content": turn.content}
            for turn in history
        ] if GroundedQA._is_contextual_followup(question) else []
        return (
            "Answer the current question using only the supplied excerpts from the selected paper. "
            "Do not use outside knowledge or invent information. Conversation history may only "
            "clarify references in the question; it is not evidence. If an excerpt directly "
            "addresses the question, answer from that excerpt and set supported to true. Set "
            "supported to false only when none of the supplied excerpts support an answer.\n"
            "Give one concise sentence that directly answers the specific aspect requested. Do not "
            "substitute related background, repeat an earlier answer, or copy a sentence that does "
            "not answer the current question.\n"
            f"Return JSON only with supported (boolean), "
            "answer (string), and evidence (array of objects containing only chunk_id). For a "
            "supported answer, select supporting chunk IDs from the supplied excerpts. Do not "
            "generate quotes or page numbers; the application attaches authoritative source text "
            "and provenance. For an unsupported answer, return supported=false and an empty "
            "evidence array.\n"
            f"CONVERSATION HISTORY:\n{json.dumps(turns, ensure_ascii=False)}\n"
            f"CURRENT QUESTION:\n{question}\n"
            f"PAPER EXCERPTS:\n{json.dumps(excerpts, ensure_ascii=False)}"
        )

    @classmethod
    def _parse_response(cls, response: str) -> tuple[bool, str, tuple[str, ...]]:
        try:
            payload: Any = json.loads(response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise MalformedQAResponse("LLM response must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise MalformedQAResponse("LLM response must be a JSON object")
        supported = payload.get("supported")
        answer = payload.get("answer")
        raw_evidence = payload.get("evidence")
        if (
            not isinstance(supported, bool)
            or not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(raw_evidence, list)
        ):
            raise MalformedQAResponse("LLM response has invalid supported, answer, or evidence fields")

        chunk_ids: list[str] = []
        try:
            for item in raw_evidence:
                if not isinstance(item, dict):
                    raise TypeError("evidence entries must be objects")
                chunk_id = item["chunk_id"]
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise TypeError("evidence chunk_id must be a non-empty string")
                chunk_ids.append(chunk_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedQAResponse(f"invalid evidence: {exc}") from exc

        if supported and not chunk_ids:
            raise MalformedQAResponse("supported answers require at least one evidence reference")
        return supported, answer.strip(), tuple(dict.fromkeys(chunk_ids))

    @staticmethod
    def _authoritative_evidence(
        chunk_ids: Sequence[str],
        chunks_by_id: dict[str, Any],
    ) -> tuple[QAEvidence, ...]:
        evidence: list[QAEvidence] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None or not chunk.page_numbers:
                continue
            evidence.append(
                QAEvidence(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_numbers[0],
                    quote=chunk.text,
                )
            )
        return tuple(evidence)

    @staticmethod
    def _validate_evidence(
        evidence: Sequence[QAEvidence],
        results: Sequence[VectorSearchResult],
    ) -> None:
        chunks = {result.chunk.chunk_id: result.chunk for result in results}
        for item in evidence:
            chunk = chunks.get(item.chunk_id)
            if (
                chunk is None
                or item.page_number not in chunk.page_numbers
                or item.quote not in chunk.text
            ):
                raise MalformedQAResponse(
                    "evidence quote, chunk ID, or page number does not match retrieved paper text"
                )

    @staticmethod
    def _extractive_fallback(
        results: Sequence[VectorSearchResult],
    ) -> tuple[str, tuple[QAEvidence, ...]] | None:
        return GroundedQA._top_chunk_fallback(results)

    @staticmethod
    def _top_chunk_fallback(
        results: Sequence[VectorSearchResult],
    ) -> tuple[str, tuple[QAEvidence, ...]] | None:
        for result in results:
            chunk = result.chunk
            if chunk.page_numbers and chunk.text.strip():
                evidence = GroundedQA._authoritative_evidence(
                    (chunk.chunk_id,), {chunk.chunk_id: chunk}
                )
                if evidence:
                    return chunk.text.strip(), evidence
        return None

    @staticmethod
    def _answer_is_source_text(
        answer: str,
        evidence: Sequence[QAEvidence],
    ) -> bool:
        normalized_answer = " ".join(answer.casefold().split())
        return any(
            normalized_answer in " ".join(item.quote.casefold().split())
            for item in evidence
        )

    @staticmethod
    def _answer_addresses_question(answer: str, question: str) -> bool:
        question_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", question.casefold())
            if len(token) > 2 and token not in _QUESTION_STOP_WORDS
        }
        answer_terms = set(re.findall(r"[a-z0-9]+", answer.casefold()))
        return bool(question_terms & answer_terms)

    @staticmethod
    def _not_found(
        question: str,
        results: Sequence[VectorSearchResult] = (),
    ) -> QAResult:
        return QAResult(
            question=question,
            answer=NOT_FOUND_ANSWER,
            source_chunks=tuple(item.chunk for item in results),
            page_numbers=(),
            retrieval_distances=tuple(item.distance for item in results),
            evidence=(),
            grounded=False,
        )
