"""Shared, checkpointed LangGraph state for one digest-agent session."""

from dataclasses import dataclass
from typing import TypedDict

from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.parsing.models import ParsedPaper
from arxiv_digest.qa.models import ConversationTurn, QAResult
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.retrieval.query import QueryClassification
from arxiv_digest.summarization.models import PaperBriefing


@dataclass(frozen=True, slots=True)
class AgentIssue:
    stage: str
    code: str
    message: str


class AgentState(TypedDict, total=False):
    original_query: str
    query_type: str
    query_interpretation: QueryClassification
    candidate_papers: list[PaperMetadata]
    selected_paper: PaperMetadata | None
    selection_reason: str | None
    parsed_paper: ParsedPaper | None
    chunks: list[PaperChunk]
    indexed_chunk_ids: list[str]
    vector_store_ref: str | None
    briefing: PaperBriefing | None
    conversation_history: list[ConversationTurn]
    current_question: str | None
    latest_qa_result: QAResult | None
    errors: list[AgentIssue]
    warnings: list[str]
    status: str
