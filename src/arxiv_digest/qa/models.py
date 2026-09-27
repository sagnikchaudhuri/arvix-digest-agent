"""Structured models for grounded question answering."""

from dataclasses import dataclass
from typing import Literal

from arxiv_digest.indexing.models import PaperChunk


NOT_FOUND_ANSWER = "The answer is not found in the available paper text."


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("conversation role must be user or assistant")
        if not self.content.strip():
            raise ValueError("conversation content cannot be empty")


@dataclass(frozen=True, slots=True)
class QAEvidence:
    chunk_id: str
    page_number: int
    quote: str

    def __post_init__(self) -> None:
        if not self.chunk_id.strip() or self.page_number < 1 or not self.quote.strip():
            raise ValueError("evidence requires a chunk ID, positive page, and quote")


@dataclass(frozen=True, slots=True)
class QAResult:
    question: str
    answer: str
    source_chunks: tuple[PaperChunk, ...]
    page_numbers: tuple[int, ...]
    retrieval_distances: tuple[float, ...]
    evidence: tuple[QAEvidence, ...]
    grounded: bool

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("question and answer cannot be empty")
        if len(self.source_chunks) != len(self.retrieval_distances):
            raise ValueError("each source chunk must have one retrieval distance")
        if self.grounded and not self.evidence:
            raise ValueError("a grounded answer must include evidence")
        if not self.grounded and self.evidence:
            raise ValueError("an ungrounded answer cannot include validated evidence")
