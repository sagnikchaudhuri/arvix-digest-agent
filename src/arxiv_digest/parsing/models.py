"""Application models for page-aware parsed paper content."""

from dataclasses import dataclass
from pathlib import Path

from arxiv_digest.retrieval.models import PaperMetadata


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class SectionHeading:
    title: str
    page_number: int


@dataclass(frozen=True, slots=True)
class ParseWarning:
    code: str
    message: str
    page_number: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedPaper:
    paper: PaperMetadata
    pdf_path: Path
    full_text: str
    pages: tuple[PageText, ...]
    abstract: str | None
    abstract_source: str | None
    sections: tuple[SectionHeading, ...]
    references: str | None
    warnings: tuple[ParseWarning, ...]
