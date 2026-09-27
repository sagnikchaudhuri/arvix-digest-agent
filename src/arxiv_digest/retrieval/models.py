"""Internal data types for arXiv paper metadata."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    title: str
    authors: tuple[str, ...]
    arxiv_id: str
    abstract: str
    published: datetime
    categories: tuple[str, ...]
    pdf_url: str
    abstract_url: str
