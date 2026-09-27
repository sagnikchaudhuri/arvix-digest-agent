"""Validated output models for grounded paper briefings."""

from dataclasses import dataclass


NOT_FOUND = "Not found in the available paper text."
CONTENT_FIELDS = ("summary", "problem", "method", "key_results", "limitations")


@dataclass(frozen=True, slots=True)
class Evidence:
    field: str
    quote: str
    page_numbers: tuple[int, ...]
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if self.field not in CONTENT_FIELDS:
            raise ValueError(f"unsupported evidence field: {self.field}")
        if not self.quote.strip() or not self.page_numbers:
            raise ValueError("evidence requires a quote and at least one page number")
        if any(page < 1 for page in self.page_numbers):
            raise ValueError("evidence page numbers must be positive")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Model-selected source IDs; quote and page provenance are application-owned."""

    field: str
    chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.field not in CONTENT_FIELDS:
            raise ValueError(f"unsupported evidence field: {self.field}")
        if any(not chunk_id.strip() for chunk_id in self.chunk_ids):
            raise ValueError("evidence references require non-empty chunk IDs")


@dataclass(frozen=True, slots=True)
class ChunkDigest:
    summary: str
    problem: str
    method: str
    key_results: str
    limitations: str
    evidence: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        for field in CONTENT_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            if value != NOT_FOUND and not any(item.field == field for item in self.evidence):
                raise ValueError(f"{field} requires source evidence")


@dataclass(frozen=True, slots=True)
class PaperBriefing:
    title: str
    authors: tuple[str, ...]
    arxiv_id: str
    publication_date: str
    arxiv_link: str
    plain_english_summary: str
    problem: str
    method: str
    key_results: str
    limitations: str
    suggested_follow_up_questions: tuple[str, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        for field in (
            "title",
            "arxiv_id",
            "publication_date",
            "arxiv_link",
            "plain_english_summary",
            "problem",
            "method",
            "key_results",
            "limitations",
        ):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} cannot be empty")
        if not self.suggested_follow_up_questions or any(
            not question.strip() for question in self.suggested_follow_up_questions
        ):
            raise ValueError("at least one non-empty follow-up question is required")
        for field, attribute in (
            ("summary", "plain_english_summary"),
            ("problem", "problem"),
            ("method", "method"),
            ("key_results", "key_results"),
            ("limitations", "limitations"),
        ):
            if getattr(self, attribute) != NOT_FOUND and not any(
                item.field == field for item in self.evidence
            ):
                raise ValueError(f"{attribute} requires source evidence")
