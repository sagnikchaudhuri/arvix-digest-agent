"""Structured chunks shared by embedding and vector storage components."""

from dataclasses import dataclass
from typing import TypeAlias


MetadataValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True, slots=True)
class PaperChunk:
    chunk_id: str
    paper_id: str
    text: str
    page_numbers: tuple[int, ...]
    section: str | None
    metadata: dict[str, MetadataValue]
