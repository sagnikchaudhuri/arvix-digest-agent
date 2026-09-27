"""Backend-neutral vector-store types and interface."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from arxiv_digest.indexing.models import PaperChunk


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    chunk: PaperChunk
    distance: float


class VectorStore(Protocol):
    def upsert(
        self,
        chunks: Sequence[PaperChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[VectorSearchResult]: ...

