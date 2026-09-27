from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
import unittest

from arxiv_digest.agent import ArxivDigestAgent, select_best_paper
from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.parsing import InsufficientTextError, InvalidPDFError, PageText, ParsedPaper, ParseWarning
from arxiv_digest.qa import QAEvidence, QAResult
from arxiv_digest.retrieval.errors import ArxivApiError
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.summarization.models import NOT_FOUND, PaperBriefing
from arxiv_digest.vector_store.store import VectorSearchResult


def make_paper(
    arxiv_id: str = "2501.12345v1",
    title: str = "KV Cache Compression for Long Context Models",
    abstract: str = "We evaluate KV cache compression and measure memory use.",
) -> PaperMetadata:
    return PaperMetadata(
        title=title,
        authors=("A. Author",),
        arxiv_id=arxiv_id,
        abstract=abstract,
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        categories=("cs.AI",),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abstract_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def make_parsed(paper: PaperMetadata) -> ParsedPaper:
    text = "KV cache compression reduces memory use in long-context models."
    return ParsedPaper(
        paper=paper,
        pdf_path=Path("fixture.pdf"),
        full_text=text,
        pages=(PageText(1, text),),
        abstract=paper.abstract,
        abstract_source="metadata",
        sections=(),
        references=None,
        warnings=(),
    )


def make_chunk(paper_id: str = "2501.12345v1") -> PaperChunk:
    return PaperChunk(
        chunk_id=f"{paper_id}:0000",
        paper_id=paper_id,
        text="KV cache compression reduces memory use in long-context models.",
        page_numbers=(1,),
        section="Introduction",
        metadata={"paper_id": paper_id},
    )


class FakeRetriever:
    def __init__(self, papers: list[PaperMetadata] | None = None, error: Exception | None = None) -> None:
        self.papers = papers if papers is not None else [make_paper()]
        self.error = error
        self.calls: list[str] = []

    def retrieve(self, query: str) -> list[PaperMetadata]:
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.papers


class FakeParser:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def parse(self, paper: PaperMetadata) -> ParsedPaper:
        self.calls.append(paper.arxiv_id)
        if self.error:
            raise self.error
        return make_parsed(paper)


class FakeChunker:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def chunk(self, parsed: ParsedPaper) -> tuple[PaperChunk, ...]:
        self.calls += 1
        if self.error:
            raise self.error
        return (make_chunk(parsed.paper.arxiv_id),)


class FakeEmbedder:
    def embed_chunks(self, chunks: tuple[PaperChunk, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _chunk in chunks)

    def embed_texts(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _text in texts)


class FakeVectorStore:
    persist_directory = "test-vector-store"

    def __init__(self) -> None:
        self.chunks: tuple[PaperChunk, ...] = ()
        self.upsert_calls = 0

    def upsert(self, chunks: tuple[PaperChunk, ...], _embeddings: object) -> None:
        self.upsert_calls += 1
        self.chunks = tuple(chunks)

    def search(
        self,
        _query_embedding: object,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(chunk, 0.1)
            for chunk in self.chunks
            if paper_id is None or chunk.paper_id == paper_id
        ][:top_k]


class FakeSummarizer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def summarize(self, paper: PaperMetadata, _parsed: ParsedPaper) -> PaperBriefing:
        self.calls += 1
        if self.error:
            raise self.error
        return PaperBriefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            publication_date=paper.published.date().isoformat(),
            arxiv_link=paper.abstract_url,
            plain_english_summary=NOT_FOUND,
            problem=NOT_FOUND,
            method=NOT_FOUND,
            key_results=NOT_FOUND,
            limitations=NOT_FOUND,
            suggested_follow_up_questions=("What should be tested next?",),
            evidence=(),
        )


class FakeQA:
    def __init__(self, store: FakeVectorStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str, tuple[object, ...], str]] = []

    def answer(
        self,
        question: str,
        paper_id: str,
        history: tuple[object, ...] = (),
        paper_context: str = "",
    ) -> QAResult:
        self.calls.append((question, paper_id, tuple(history), paper_context))
        chunk = self.store.chunks[0]
        answer = f"Answer {len(self.calls)} from the selected paper."
        return QAResult(
            question=question,
            answer=answer,
            source_chunks=(chunk,),
            page_numbers=(1,),
            retrieval_distances=(0.1,),
            evidence=(QAEvidence(chunk.chunk_id, 1, chunk.text),),
            grounded=True,
        )


class TestArxivDigestAgent(unittest.TestCase):
    def create_agent(
        self,
        *,
        papers: list[PaperMetadata] | None = None,
        retrieval_error: Exception | None = None,
        parse_error: Exception | None = None,
        chunk_error: Exception | None = None,
        summary_error: Exception | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[ArxivDigestAgent, dict[str, object]]:
        retriever = FakeRetriever(papers, retrieval_error)
        parser = FakeParser(parse_error)
        chunker = FakeChunker(chunk_error)
        store = FakeVectorStore()
        summarizer = FakeSummarizer(summary_error)
        qa = FakeQA(store)
        agent = ArxivDigestAgent(
            retriever=retriever,
            pdf_parser=parser,
            chunker=chunker,
            embedder=FakeEmbedder(),
            vector_store=store,
            summarizer=summarizer,
            qa_service=qa,
            progress_callback=progress_callback,
        )
        return agent, {
            "retriever": retriever,
            "parser": parser,
            "chunker": chunker,
            "store": store,
            "summarizer": summarizer,
            "qa": qa,
        }

    def test_topic_query_runs_full_graph_and_preserves_shared_state(self) -> None:
        agent, services = self.create_agent()
        state = agent.run("KV cache compression")

        self.assertEqual(state["status"], "briefing_ready")
        self.assertEqual(state["query_type"], "topic")
        self.assertEqual(state["selected_paper"].arxiv_id, "2501.12345v1")
        self.assertIsNotNone(state["parsed_paper"])
        self.assertEqual(state["indexed_chunk_ids"], ["2501.12345v1:0000"])
        self.assertEqual(state["vector_store_ref"], "test-vector-store")
        self.assertIsNotNone(state["briefing"])
        self.assertEqual(services["summarizer"].calls, 1)

    def test_direct_arxiv_id_and_url_bypass_topic_selection(self) -> None:
        for query in ("2501.12345v1", "https://arxiv.org/abs/2501.12345v1"):
            with self.subTest(query=query):
                agent, _ = self.create_agent()
                state = agent.run(query)
                self.assertEqual(state["status"], "briefing_ready")
                self.assertIn("Direct arXiv", state["selection_reason"])
                self.assertEqual(state["query_type"], "paper_id" if "://" not in query else "paper_url")

    def test_multiple_candidates_are_ranked_deterministically(self) -> None:
        unrelated = make_paper(
            "2502.11111v1", "Language Model Safety Evaluation", "Safety and alignment benchmarks."
        )
        relevant = make_paper(
            "2503.22222v1", "KV Cache Compression", "Compression reduces long-context memory use."
        )
        agent, _ = self.create_agent(papers=[unrelated, relevant])
        state = agent.run("KV cache compression")
        self.assertEqual(state["selected_paper"].arxiv_id, relevant.arxiv_id)
        self.assertEqual(select_best_paper([unrelated, relevant], "KV cache compression")[0], relevant)

    def test_zero_candidates_terminate_with_structured_issue(self) -> None:
        agent, services = self.create_agent(papers=[])
        state = agent.run("KV cache compression")
        self.assertEqual(state["status"], "no_candidates")
        self.assertEqual(state["errors"][0].code, "no_candidates")
        self.assertEqual(services["parser"].calls, [])

    def test_invalid_query_does_not_call_retriever(self) -> None:
        agent, services = self.create_agent()
        state = agent.run("2301.not-an-id")
        self.assertEqual(state["status"], "invalid_query")
        self.assertEqual(state["errors"][0].stage, "understand_query")
        self.assertEqual(services["retriever"].calls, [])

    def test_retrieval_failure_terminates_with_structured_error(self) -> None:
        agent, services = self.create_agent(retrieval_error=ArxivApiError("offline"))
        state = agent.run("KV cache compression")
        self.assertEqual(state["status"], "retrieval_failed")
        self.assertEqual(state["errors"][0].code, "ArxivApiError")
        self.assertEqual(services["parser"].calls, [])

    def test_pdf_failure_and_insufficient_text_have_distinct_routes(self) -> None:
        agent, _ = self.create_agent(
            parse_error=InvalidPDFError("2501.12345v1", "Invalid PDF fixture.")
        )
        failed = agent.run("KV cache compression")
        self.assertEqual(failed["status"], "pdf_failed")
        self.assertEqual(failed["errors"][0].code, "invalid_pdf")

        insufficient = InsufficientTextError(
            "2501.12345v1",
            "No extractable text.",
            warnings=(ParseWarning("ocr_unsupported", "OCR is not performed."),),
        )
        agent, _ = self.create_agent(parse_error=insufficient)
        state = agent.run("KV cache compression")
        self.assertEqual(state["status"], "insufficient_text")
        self.assertEqual(state["warnings"], ["OCR is not performed."])

    def test_indexing_failure_terminates_before_summary(self) -> None:
        agent, services = self.create_agent(chunk_error=RuntimeError("chunk failure"))
        state = agent.run("KV cache compression")
        self.assertEqual(state["status"], "indexing_failed")
        self.assertEqual(state["errors"][0].stage, "index_paper")
        self.assertEqual(services["summarizer"].calls, 0)

    def test_summarization_failure_is_structured(self) -> None:
        progress: list[str] = []
        agent, _ = self.create_agent(
            summary_error=RuntimeError("summary failure"),
            progress_callback=progress.append,
        )
        state = agent.run("KV cache compression")
        self.assertEqual(state["status"], "summarization_failed")
        self.assertEqual(state["errors"][0].stage, "summarize_paper")
        self.assertEqual(progress, ["summarization_started", "summarization_failed"])

    def test_summarization_progress_callback_reports_start_and_success(self) -> None:
        progress: list[str] = []
        agent, _ = self.create_agent(progress_callback=progress.append)

        state = agent.run("KV cache compression")

        self.assertEqual(state["status"], "briefing_ready")
        self.assertEqual(progress, ["summarization_started", "summarization_completed"])

    def test_multiple_qa_turns_preserve_history_without_reingestion(self) -> None:
        agent, services = self.create_agent()
        agent.run("KV cache compression")
        first = agent.ask("What does the method compress?")
        second = agent.ask("What is the stated benefit?")
        state = agent.state

        self.assertTrue(first.grounded and second.grounded)
        self.assertEqual(len(services["qa"].calls), 2)
        self.assertEqual(len(services["qa"].calls[0][2]), 0)
        self.assertEqual(len(services["qa"].calls[1][2]), 2)
        self.assertIn("KV Cache Compression", services["qa"].calls[0][3])
        self.assertEqual(len(state["conversation_history"]), 4)
        self.assertEqual(state["selected_paper"].arxiv_id, "2501.12345v1")
        self.assertEqual(state["indexed_chunk_ids"], ["2501.12345v1:0000"])
        self.assertEqual(state["chunks"][0].paper_id, "2501.12345v1")
        self.assertEqual(state["vector_store_ref"], "test-vector-store")
        self.assertEqual(state["status"], "qa_answered")
        self.assertEqual(len(services["retriever"].calls), 1)
        self.assertEqual(len(services["parser"].calls), 1)
        self.assertEqual(services["chunker"].calls, 1)
        self.assertEqual(services["store"].upsert_calls, 1)
        self.assertEqual(services["summarizer"].calls, 1)

    def test_reset_discards_checkpointed_session(self) -> None:
        agent, _ = self.create_agent()
        agent.run("KV cache compression")
        self.assertIsNotNone(agent.state)
        agent.reset()
        self.assertIsNone(agent.state)
        with self.assertRaises(RuntimeError):
            agent.ask("A follow-up")


if __name__ == "__main__":
    unittest.main()
