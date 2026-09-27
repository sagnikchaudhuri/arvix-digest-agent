"""Explicit LangGraph orchestration for the arXiv digest and QA pipeline."""

from collections.abc import Callable, Sequence
import re
from typing import Any, cast
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from arxiv_digest.indexing import Embedder, SentenceTransformerEmbedder, TextChunker
from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.parsing import InsufficientTextError, PDFParser, PDFProcessingError, ParsedPaper
from arxiv_digest.parsing.models import PageText, ParseWarning, SectionHeading
from arxiv_digest.qa import ConversationTurn, GroundedQA, QAResult
from arxiv_digest.qa.models import QAEvidence
from arxiv_digest.retrieval import ArxivRetriever
from arxiv_digest.retrieval.errors import NoResultsError
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.retrieval.query import QueryClassification, QueryKind, classify_query
from arxiv_digest.summarization import PaperBriefing, PaperSummarizer
from arxiv_digest.summarization.models import Evidence
from arxiv_digest.vector_store import ChromaVectorStore, VectorStore

from arxiv_digest.agent.state import AgentIssue, AgentState


_STOP_WORDS = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "with", "work", "recent",
    "research", "paper", "papers", "about",
}

_CHECKPOINT_TYPES = (
    AgentIssue,
    PaperChunk,
    PageText,
    ParseWarning,
    ParsedPaper,
    SectionHeading,
    ConversationTurn,
    QAEvidence,
    QAResult,
    PaperMetadata,
    QueryClassification,
    QueryKind,
    PaperBriefing,
    Evidence,
)


def select_best_paper(
    candidates: Sequence[PaperMetadata],
    query: str,
) -> tuple[PaperMetadata, str]:
    """Choose deterministically using title/abstract overlap, metadata, then API order."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    terms = set(re.findall(r"[a-z0-9]+", query.casefold())) - _STOP_WORDS

    def rank(item: tuple[int, PaperMetadata]) -> tuple[float, int, int]:
        index, paper = item
        title_terms = set(re.findall(r"[a-z0-9]+", paper.title.casefold()))
        abstract_terms = set(re.findall(r"[a-z0-9]+", paper.abstract.casefold()))
        title_overlap = len(terms & title_terms) / max(len(terms), 1)
        abstract_overlap = len(terms & abstract_terms) / max(len(terms), 1)
        relevance = 2 * title_overlap + abstract_overlap
        metadata_quality = sum(
            bool(value)
            for value in (paper.title.strip(), paper.abstract.strip(), paper.authors, paper.categories)
        )
        return relevance, metadata_quality, -index

    winner_index, winner = max(enumerate(candidates), key=rank)
    reason = (
        "Ranked by query-token overlap (title weighted twice abstract), then metadata "
        f"completeness, then arXiv result order; selected candidate {winner_index + 1}."
    )
    return winner, reason


class ArxivDigestAgent:
    """Stateful research/briefing graph with checkpointed follow-up QA turns."""

    def __init__(
        self,
        *,
        retriever: Any | None = None,
        pdf_parser: Any | None = None,
        chunker: Any | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        summarizer: Any | None = None,
        qa_service: Any | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.retriever = retriever or ArxivRetriever()
        self.pdf_parser = pdf_parser or PDFParser()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.summarizer = summarizer or PaperSummarizer()
        self.qa_service = qa_service or GroundedQA(self.embedder, self.vector_store)
        self._progress_callback = progress_callback
        self._checkpointer = checkpointer or InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
        )
        self._thread_id: str | None = None
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("understand_query", self._understand_query)
        builder.add_node("retrieve_papers", self._retrieve_papers)
        builder.add_node("select_paper", self._select_paper)
        builder.add_node("fetch_and_parse", self._fetch_and_parse)
        builder.add_node("index_paper", self._index_paper)
        builder.add_node("summarize_paper", self._summarize_paper)
        builder.add_node("answer_question", self._answer_question)

        builder.add_conditional_edges(
            START,
            self._route_entry,
            {"research": "understand_query", "qa": "answer_question"},
        )
        builder.add_conditional_edges(
            "understand_query",
            self._route_after_understanding,
            {"retrieve": "retrieve_papers", "stop": END},
        )
        builder.add_conditional_edges(
            "retrieve_papers",
            self._route_after_retrieval,
            {
                "select": "select_paper",
                "direct": "fetch_and_parse",
                "stop": END,
            },
        )
        builder.add_edge("select_paper", "fetch_and_parse")
        builder.add_conditional_edges(
            "fetch_and_parse",
            self._route_after_parsing,
            {"index": "index_paper", "stop": END},
        )
        builder.add_conditional_edges(
            "index_paper",
            self._route_after_indexing,
            {"summarize": "summarize_paper", "stop": END},
        )
        builder.add_conditional_edges(
            "summarize_paper",
            self._route_after_summarization,
            {"ready": END, "stop": END},
        )
        builder.add_edge("answer_question", END)
        return builder.compile(checkpointer=self._checkpointer)

    def run(self, query: str) -> AgentState:
        """Start a fresh research pipeline and return its final shared state."""
        self.reset()
        self._thread_id = uuid4().hex
        initial: AgentState = {
            "original_query": query,
            "candidate_papers": [],
            "selected_paper": None,
            "selection_reason": None,
            "parsed_paper": None,
            "chunks": [],
            "indexed_chunk_ids": [],
            "vector_store_ref": None,
            "briefing": None,
            "conversation_history": [],
            "current_question": None,
            "latest_qa_result": None,
            "errors": [],
            "warnings": [],
            "status": "running",
        }
        self.graph.invoke(initial, self._config())
        return self.state or cast(AgentState, initial)

    def ask(self, question: str) -> QAResult:
        """Ask a follow-up against the active paper without rerunning ingestion."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        current = self.state
        if (
            current is None
            or current.get("status") not in {"briefing_ready", "qa_answered", "qa_failed"}
            or current.get("selected_paper") is None
        ):
            raise RuntimeError("run a successful paper briefing before asking follow-up questions")
        self.graph.invoke({"current_question": question}, self._config())
        latest = self.state
        result = latest.get("latest_qa_result") if latest else None
        if result is None:
            raise RuntimeError("QA graph did not produce a result")
        return result

    def reset(self) -> None:
        """Delete the active checkpointed conversation and clear this session."""
        if self._thread_id is not None:
            self._checkpointer.delete_thread(self._thread_id)
        self._thread_id = None

    @property
    def state(self) -> AgentState | None:
        if self._thread_id is None:
            return None
        snapshot = self.graph.get_state(self._config())
        return cast(AgentState, snapshot.values) if snapshot.values else None

    def _config(self) -> dict[str, dict[str, str]]:
        if self._thread_id is None:
            raise RuntimeError("agent has no active graph thread")
        return {"configurable": {"thread_id": self._thread_id}}

    @staticmethod
    def _route_entry(state: AgentState) -> str:
        if state.get("current_question") and state.get("selected_paper") is not None:
            return "qa"
        return "research"

    def _understand_query(self, state: AgentState) -> dict[str, Any]:
        classification = classify_query(state.get("original_query"))
        update: dict[str, Any] = {
            "query_interpretation": classification,
            "query_type": classification.kind.value,
        }
        if classification.kind is QueryKind.INVALID:
            update.update(
                status="invalid_query",
                errors=(AgentIssue("understand_query", "invalid_query", classification.message or "Invalid query."),),
            )
        return update

    @staticmethod
    def _route_after_understanding(state: AgentState) -> str:
        return "stop" if state.get("status") == "invalid_query" else "retrieve"

    def _retrieve_papers(self, state: AgentState) -> dict[str, Any]:
        try:
            papers = tuple(self.retriever.retrieve(state.get("original_query", "")))
        except NoResultsError as exc:
            return {
                "candidate_papers": [],
                "status": "no_candidates",
                "errors": [AgentIssue("retrieve_papers", "no_candidates", str(exc))],
            }
        except Exception as exc:
            return {
                "candidate_papers": [],
                "status": "retrieval_failed",
                "errors": [AgentIssue("retrieve_papers", type(exc).__name__, str(exc))],
            }

        if not papers:
            return {
                "candidate_papers": [],
                "status": "no_candidates",
                "errors": [AgentIssue("retrieve_papers", "no_candidates", "No arXiv candidates found.")],
            }

        classification = state.get("query_interpretation")
        direct = classification is not None and classification.kind in (
            QueryKind.PAPER_ID,
            QueryKind.PAPER_URL,
        )
        return {
            "candidate_papers": list(papers),
            "selected_paper": papers[0] if direct else None,
            "selection_reason": "Direct arXiv ID/URL lookup; no topic ranking required." if direct else None,
            "status": "paper_selected" if direct else "candidates_retrieved",
        }

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        if state.get("status") == "candidates_retrieved":
            return "select"
        if state.get("status") == "paper_selected":
            return "direct"
        return "stop"

    @staticmethod
    def _select_paper(state: AgentState) -> dict[str, Any]:
        candidates = state.get("candidate_papers", ())
        if not candidates:
            return {
                "status": "no_candidates",
                "errors": [AgentIssue("select_paper", "no_candidates", "No candidates were available to rank.")],
            }
        paper, reason = select_best_paper(candidates, state.get("original_query", ""))
        return {"selected_paper": paper, "selection_reason": reason, "status": "paper_selected"}

    def _fetch_and_parse(self, state: AgentState) -> dict[str, Any]:
        paper = state.get("selected_paper")
        if paper is None:
            return {
                "status": "pdf_failed",
                "errors": [AgentIssue("fetch_and_parse", "missing_paper", "No paper was selected.")],
            }
        try:
            parsed: ParsedPaper = self.pdf_parser.parse(paper)
        except InsufficientTextError as exc:
            warnings = [warning.message for warning in exc.warnings]
            return {
                "status": "insufficient_text",
                "warnings": warnings,
                "errors": [AgentIssue("fetch_and_parse", exc.code, exc.message)],
            }
        except PDFProcessingError as exc:
            warnings = [warning.message for warning in exc.warnings]
            return {
                "status": "pdf_failed",
                "warnings": warnings,
                "errors": [AgentIssue("fetch_and_parse", exc.code, exc.message)],
            }
        except Exception as exc:
            return {
                "status": "pdf_failed",
                "errors": [AgentIssue("fetch_and_parse", type(exc).__name__, str(exc))],
            }
        return {
            "parsed_paper": parsed,
            "warnings": [warning.message for warning in parsed.warnings],
            "status": "parsed",
        }

    @staticmethod
    def _route_after_parsing(state: AgentState) -> str:
        return "index" if state.get("status") == "parsed" else "stop"

    def _index_paper(self, state: AgentState) -> dict[str, Any]:
        parsed = state.get("parsed_paper")
        if parsed is None:
            return {
                "status": "indexing_failed",
                "errors": [AgentIssue("index_paper", "missing_parsed_paper", "Parsed paper is unavailable.")],
            }
        try:
            chunks: tuple[PaperChunk, ...] = self.chunker.chunk(parsed)
            if not chunks:
                return {
                    "status": "insufficient_text",
                    "errors": [AgentIssue("index_paper", "no_indexable_chunks", "No text chunks were produced.")],
                }
            vectors = self.embedder.embed_chunks(chunks)
            self.vector_store.upsert(chunks, vectors)
        except Exception as exc:
            return {
                "status": "indexing_failed",
                "errors": [AgentIssue("index_paper", type(exc).__name__, str(exc))],
            }

        store_ref = getattr(self.vector_store, "persist_directory", None)
        return {
            "chunks": list(chunks),
            "indexed_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "vector_store_ref": str(store_ref) if store_ref is not None else "injected-vector-store",
            "status": "indexed",
        }

    @staticmethod
    def _route_after_indexing(state: AgentState) -> str:
        return "summarize" if state.get("status") == "indexed" else "stop"

    def _summarize_paper(self, state: AgentState) -> dict[str, Any]:
        paper, parsed = state.get("selected_paper"), state.get("parsed_paper")
        if paper is None or parsed is None:
            return {
                "status": "summarization_failed",
                "errors": [AgentIssue("summarize_paper", "missing_input", "Paper metadata or parsed text is unavailable.")],
            }
        if self._progress_callback is not None:
            self._progress_callback("summarization_started")
        try:
            briefing: PaperBriefing = self.summarizer.summarize(paper, parsed)
        except Exception as exc:
            if self._progress_callback is not None:
                self._progress_callback("summarization_failed")
            return {
                "status": "summarization_failed",
                "errors": [AgentIssue("summarize_paper", type(exc).__name__, str(exc))],
            }
        if self._progress_callback is not None:
            self._progress_callback("summarization_completed")
        return {"briefing": briefing, "status": "briefing_ready"}

    @staticmethod
    def _route_after_summarization(state: AgentState) -> str:
        return "ready" if state.get("status") == "briefing_ready" else "stop"

    def _answer_question(self, state: AgentState) -> dict[str, Any]:
        question = state.get("current_question")
        paper = state.get("selected_paper")
        if not question or paper is None:
            return {
                "status": "qa_failed",
                "current_question": None,
                "latest_qa_result": None,
                "errors": [AgentIssue("answer_question", "missing_input", "Question or selected paper is unavailable.")],
            }
        history = state.get("conversation_history", ())
        try:
            result: QAResult = self.qa_service.answer(
                question,
                paper.arxiv_id,
                history=history,
                paper_context=f"{paper.title}\n{paper.abstract}",
            )
        except Exception as exc:
            return {
                "status": "qa_failed",
                "current_question": None,
                "latest_qa_result": None,
                "errors": [AgentIssue("answer_question", type(exc).__name__, str(exc))],
            }
        updated_history = (
            *history,
            ConversationTurn("user", question),
            ConversationTurn("assistant", result.answer),
        )
        return {
            "latest_qa_result": result,
            "conversation_history": list(updated_history),
            "current_question": None,
            "status": "qa_answered",
            "errors": [],
        }
