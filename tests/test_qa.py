import json
from tempfile import TemporaryDirectory
import unittest

from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.qa import (
    NOT_FOUND_ANSWER,
    ConversationTurn,
    GroundedQA,
    MalformedQAResponse,
)
from arxiv_digest.vector_store import ChromaVectorStore
from arxiv_digest.vector_store.store import VectorSearchResult


def make_chunk(chunk_id: str = "chunk-a", paper_id: str = "paper-a") -> PaperChunk:
    text = "Dataset ABC contained 1200 samples."
    return PaperChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        page_numbers=(4, 5),
        section="Experiments",
        metadata={"paper_id": paper_id},
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        self.texts.extend(texts)
        return tuple((0.25, 0.75) for _ in texts)

    def embed_chunks(self, _chunks: object) -> tuple[tuple[float, ...], ...]:
        raise AssertionError("QA should embed the question, not paper chunks")


class FakeVectorStore:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[float, ...], int, str | None]] = []

    def search(
        self,
        query_embedding: tuple[float, ...],
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append((tuple(query_embedding), top_k, paper_id))
        # Deliberately return all results; the QA service must also enforce paper ID.
        return self.results[:top_k]

    def upsert(self, _chunks: object, _embeddings: object) -> None:
        raise AssertionError("QA should not write to the vector store")


class FakeProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def supported_response(chunk: PaperChunk | None = None) -> str:
    chunk = chunk or make_chunk()
    return json.dumps(
        {
            "supported": True,
            "answer": "The dataset contained 1200 samples.",
            "evidence": [{"chunk_id": chunk.chunk_id}],
        }
    )


class TestGroundedQA(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = make_chunk()
        self.embedder = FakeEmbedder()
        self.store = FakeVectorStore([VectorSearchResult(self.chunk, 0.12)])
        self.provider = FakeProvider(supported_response(self.chunk))
        self.qa = GroundedQA(self.embedder, self.store, self.provider, top_k=3)

    def test_successful_answer_is_grounded_and_cited(self) -> None:
        result = self.qa.answer("What dataset did they use?", "paper-a")

        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, self.chunk.text)
        self.assertEqual(result.source_chunks[0].chunk_id, "chunk-a")
        self.assertEqual(result.evidence[0].chunk_id, "chunk-a")
        self.assertEqual(result.evidence[0].page_number, 4)

    def test_retrieval_is_filtered_to_selected_paper_and_top_k(self) -> None:
        other = make_chunk("chunk-b", "paper-b")
        store = FakeVectorStore(
            [VectorSearchResult(other, 0.01), VectorSearchResult(self.chunk, 0.2)]
        )
        result = GroundedQA(
            self.embedder, store, self.provider, top_k=2
        ).answer("What dataset?", "paper-a")

        self.assertEqual(store.calls[0], ((0.25, 0.75), 2, "paper-a"))
        self.assertEqual([chunk.paper_id for chunk in result.source_chunks], ["paper-a"])

    def test_paper_context_fuses_natural_question_retrieval_and_keeps_filter(self) -> None:
        relevant = make_chunk("relevant", "paper-a")
        contextual = PaperChunk(
            "contextual", "paper-a", "The paper studies the main physical question.",
            (2,), "Introduction", {"paper_id": "paper-a"},
        )
        unrelated_paper = make_chunk("foreign", "paper-b")

        class ContextEmbedder(FakeEmbedder):
            def embed_texts(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
                self.texts.extend(texts)
                return ((1.0, 0.0), (0.0, 1.0))

        class RankedStore(FakeVectorStore):
            def search(self, query_embedding, top_k=5, paper_id=None):
                self.calls.append((tuple(query_embedding), top_k, paper_id))
                candidates = (
                    [VectorSearchResult(relevant, 0.81), VectorSearchResult(unrelated_paper, 0.1)]
                    if query_embedding[0] == 1.0
                    else [VectorSearchResult(contextual, 0.02), VectorSearchResult(relevant, 0.4)]
                )
                return candidates[:top_k]

        embedder = ContextEmbedder()
        store = RankedStore([])
        result = GroundedQA(
            embedder, store, FakeProvider(supported_response(relevant)), top_k=2
        ).answer(
            "What problem does the paper address?",
            "paper-a",
            paper_context="A Paper About Cache Compression. Cache compression reduces memory use.",
        )

        self.assertTrue(result.grounded)
        self.assertGreaterEqual(len(embedder.texts), 2)
        self.assertIn("A Paper About Cache Compression", embedder.texts[1])
        self.assertEqual([call[2] for call in store.calls], ["paper-a", "paper-a"])
        self.assertEqual({chunk.paper_id for chunk in result.source_chunks}, {"paper-a"})
        self.assertIn("relevant", {chunk.chunk_id for chunk in result.source_chunks})

    def test_fusion_prioritizes_direct_question_rank_over_context_only_rank(self) -> None:
        direct = make_chunk("direct", "paper-a")
        context_only = make_chunk("context-only", "paper-a")

        merged = GroundedQA._merge_results(
            [
                [VectorSearchResult(direct, 0.3)],
                [VectorSearchResult(context_only, 0.02)],
            ],
            "paper-a",
            top_k=1,
        )

        self.assertEqual([item.chunk.chunk_id for item in merged], ["direct"])

    def test_direct_cosine_relevance_gate_rejects_context_only_match(self) -> None:
        contextual = make_chunk("abstract-match", "paper-a")

        class TwoQueryEmbedder(FakeEmbedder):
            def embed_texts(self, texts):
                self.texts.extend(texts)
                return ((1.0, 0.0), (0.0, 1.0))

        class ContextOnlyStore(FakeVectorStore):
            def search(self, query_embedding, top_k=5, paper_id=None):
                self.calls.append((tuple(query_embedding), top_k, paper_id))
                distance = 0.814 if query_embedding[0] else 0.02
                return [VectorSearchResult(contextual, distance)]

        provider = FakeProvider(supported_response(contextual))
        result = GroundedQA(
            TwoQueryEmbedder(),
            ContextOnlyStore([]),
            provider,
            max_cosine_distance=0.81,
        ).answer(
            "What is the capital of France?",
            "paper-a",
            paper_context="The selected paper's title and abstract.",
        )

        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(provider.prompts, [])

    def test_relevant_direct_cosine_result_passes_configurable_gate(self) -> None:
        self.store.results = [VectorSearchResult(self.chunk, 0.806)]
        result = GroundedQA(
            self.embedder,
            self.store,
            self.provider,
            max_cosine_distance=0.81,
        ).answer("What dataset did they use?", "paper-a")

        self.assertTrue(result.grounded)

    def test_irrelevant_prior_questions_do_not_contaminate_query_embedding(self) -> None:
        history = (
            ConversationTurn("user", "What problem does this paper address?"),
            ConversationTurn("assistant", "It studies charge-dependent gravity."),
        )
        query = GroundedQA._retrieval_query("What is the capital of France?", history)

        self.assertEqual(query, "Current question: What is the capital of France?")

        self.qa.answer("What is the capital of France?", "paper-a", history=history)
        self.assertNotIn("charge-dependent gravity", self.provider.prompts[-1])

    def test_natural_language_question_retrieves_grounded_chunk_from_chroma(self) -> None:
        class TinySemanticEmbedder:
            @staticmethod
            def _encode(text: str) -> tuple[float, ...]:
                text = text.casefold()
                return (
                    float(any(term in text for term in ("gravity", "charge", "electro"))),
                    float(any(term in text for term in ("limitation", "scope", "classical"))),
                    float(any(term in text for term in ("france", "paris"))),
                    float(any(term in text for term in ("paper", "problem", "address"))),
                )

            def embed_texts(self, texts):
                return tuple(self._encode(text) for text in texts)

            def embed_chunks(self, chunks):
                return tuple(self._encode(chunk.text) for chunk in chunks)

        intro = PaperChunk(
            "paper-a-intro", "paper-a",
            "This paper addresses whether gravity depends on electric charge.",
            (1,), "Introduction", {"paper_id": "paper-a"},
        )
        limitations = PaperChunk(
            "paper-a-limitations", "paper-a",
            "Scope and limitations: only classical weak-field gravity is studied.",
            (4,), "Limitations", {"paper_id": "paper-a"},
        )
        foreign = PaperChunk(
            "paper-b-fact", "paper-b", "Paris is the capital of France.",
            (2,), "Background", {"paper_id": "paper-b"},
        )
        embedder = TinySemanticEmbedder()
        provider = FakeProvider(json.dumps({
            "supported": True,
            "answer": "The paper asks whether gravity depends on electric charge.",
            "evidence": [{"chunk_id": intro.chunk_id}],
        }))
        with TemporaryDirectory() as temporary_directory:
            with ChromaVectorStore(temporary_directory, "natural_qa") as store:
                chunks = (intro, limitations, foreign)
                store.upsert(chunks, embedder.embed_chunks(chunks))
                result = GroundedQA(embedder, store, provider, top_k=2).answer(
                    "What problem does this paper address?",
                    "paper-a",
                    paper_context="Does gravity care about electric charge? The paper tests charged matter.",
                )

        self.assertTrue(result.grounded)
        self.assertIn("paper-a-intro", {item.chunk_id for item in result.evidence})
        self.assertTrue(all(chunk.paper_id == "paper-a" for chunk in result.source_chunks))
        self.assertEqual(result.evidence[0].page_number, 1)
        self.assertEqual(result.evidence[0].quote, intro.text)

    def test_capital_of_france_remains_unsupported(self) -> None:
        self.store.results = [VectorSearchResult(self.chunk, 0.814)]
        self.provider.response = json.dumps(
            {"supported": False, "answer": NOT_FOUND_ANSWER, "evidence": []}
        )
        result = self.qa.answer(
            "What is the capital of France?",
            "paper-a",
            paper_context="A Paper About Cache Compression. Cache storage is studied.",
        )
        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(result.evidence, ())
        self.assertEqual(self.provider.prompts, [])

    def test_prompt_answers_when_excerpts_directly_supports_question(self) -> None:
        self.qa.answer("What dataset did they use?", "paper-a")

        prompt = self.provider.prompts[0]
        self.assertIn("If an excerpt directly addresses the question", prompt)
        self.assertIn("only when none of the supplied excerpts support", prompt)
        self.assertIn("Do not use outside knowledge", prompt)

    def test_page_numbers_and_retrieval_metadata_are_preserved(self) -> None:
        result = self.qa.answer("How large was the dataset?", "paper-a")
        self.assertEqual(result.page_numbers, (4,))
        self.assertEqual(result.source_chunks[0].page_numbers, (4, 5))
        self.assertEqual(result.retrieval_distances, (0.12,))

    def test_unsupported_question_returns_fixed_fallback(self) -> None:
        self.store.results = [VectorSearchResult(self.chunk, 0.9)]
        self.provider.response = json.dumps(
            {"supported": False, "answer": "I do not know.", "evidence": []}
        )
        result = self.qa.answer("What color was the building?", "paper-a")
        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(result.evidence, ())

    def test_model_refusal_uses_only_a_semantically_relevant_source_sentence(self) -> None:
        class DatasetEmbedder(FakeEmbedder):
            def embed_texts(self, texts):
                self.texts.extend(texts)
                return tuple(
                    (1.0, 0.0) if "dataset" in text.casefold() else (0.0, 1.0)
                    for text in texts
                )

        self.provider.response = json.dumps(
            {"supported": False, "answer": NOT_FOUND_ANSWER, "evidence": []}
        )
        embedder = DatasetEmbedder()
        result = GroundedQA(
            embedder, self.store, self.provider, max_cosine_distance=0.81
        ).answer("What dataset did they use?", "paper-a")

        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, self.chunk.text)
        self.assertEqual(result.evidence[0].quote, self.chunk.text)
        self.assertEqual(result.evidence[0].page_number, 4)

    def test_model_quote_and_page_are_ignored_in_favor_of_authoritative_chunk(self) -> None:
        self.provider.response = json.dumps(
            {
                "supported": True,
                "answer": "The dataset contained 1200 samples.",
                "evidence": [
                    {
                        "chunk_id": "chunk-a",
                        "page_number": 999,
                        "quote": "Invented quote.",
                    }
                ],
            }
        )
        result = self.qa.answer("What dataset?", "paper-a")

        self.assertTrue(result.grounded)
        self.assertEqual(result.evidence[0].chunk_id, "chunk-a")
        self.assertEqual(result.evidence[0].page_number, 4)
        self.assertEqual(result.evidence[0].quote, self.chunk.text)

    def test_nonexistent_evidence_chunk_is_rejected_as_unsupported(self) -> None:
        self.provider.response = json.dumps(
            {
                "supported": True,
                "answer": "An unsupported claim.",
                "evidence": [{"chunk_id": "not-retrieved"}],
            }
        )
        result = self.qa.answer("What dataset?", "paper-a")

        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(result.evidence, ())

    def test_exact_but_off_topic_model_sentence_uses_direct_source_chunk(self) -> None:
        self.provider.response = json.dumps(
            {
                "supported": True,
                "answer": "Theoretical implications are equally sharp.",
                "evidence": [{"chunk_id": self.chunk.chunk_id}],
            }
        )
        result = self.qa.answer("What limitations do the authors identify?", "paper-a")

        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, self.chunk.text)
        self.assertEqual(result.evidence[0].quote, self.chunk.text)

    def test_follow_up_question_receives_conversation_history(self) -> None:
        history = (
            ConversationTurn("user", "What dataset did they use?"),
            ConversationTurn("assistant", "Dataset ABC."),
        )
        result = self.qa.answer("How large was it?", "paper-a", history=history)
        self.assertTrue(result.grounded)
        self.assertIn("What dataset did they use?", self.provider.prompts[0])
        self.assertIn("How large was it?", self.provider.prompts[0])
        self.assertIn("history may only clarify references", self.provider.prompts[0])
        self.assertIn("What dataset did they use?", self.embedder.texts[0])
        self.assertIn("How large was it?", self.embedder.texts[0])
        self.assertIn("Previous answer for reference: Dataset ABC.", self.embedder.texts[0])

    def test_malformed_llm_response_is_rejected(self) -> None:
        self.provider.response = "{not json"
        with self.assertRaises(MalformedQAResponse):
            self.qa.answer("What dataset?", "paper-a")

    def test_empty_retrieval_returns_fallback_without_calling_llm(self) -> None:
        provider = FakeProvider("")
        qa = GroundedQA(self.embedder, FakeVectorStore([]), provider)
        result = qa.answer("What dataset?", "paper-a")
        self.assertFalse(result.grounded)
        self.assertEqual(result.answer, NOT_FOUND_ANSWER)
        self.assertEqual(provider.prompts, [])


if __name__ == "__main__":
    unittest.main()
