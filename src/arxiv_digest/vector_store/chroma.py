"""Chroma implementation of the backend-neutral vector-store interface."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings

from arxiv_digest.config.settings import DEFAULT_VECTOR_STORE_DIR
from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.vector_store.store import VectorSearchResult


DEFAULT_COLLECTION_NAME = "paper_chunks"


class ChromaVectorStore:
    """Persist and search paper chunks in a local Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")
        if persist_directory is None:
            from arxiv_digest.config.settings import Settings as AppSettings

            persist_directory = AppSettings.from_environment().vector_store_dir
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def close(self) -> None:
        """Release the persistent client's database resources."""
        self._client.close()

    def __enter__(self) -> "ChromaVectorStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def upsert(
        self,
        chunks: Sequence[PaperChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("each chunk must have exactly one embedding")
        if not chunks:
            return

        vectors = [self._validate_vector(embedding) for embedding in embeddings]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("all embeddings in one upsert must have the same dimension")

        ids_by_paper: dict[str, set[str]] = {}
        for chunk in chunks:
            ids_by_paper.setdefault(chunk.paper_id, set()).add(chunk.chunk_id)
        for paper_id, current_ids in ids_by_paper.items():
            stored = self._collection.get(
                where={"paper_id": paper_id},
                include=[],
            )
            stale_ids = [chunk_id for chunk_id in stored["ids"] if chunk_id not in current_ids]
            if stale_ids:
                self._collection.delete(ids=stale_ids)

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self._validate_vector(query_embedding)
        query: dict[str, object] = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if paper_id is not None:
            query["where"] = {"paper_id": paper_id}

        response = self._collection.query(**query)
        ids = response["ids"][0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        results: list[VectorSearchResult] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if text is None or metadata is None or distance is None:
                continue
            chunk = self._chunk_from_result(chunk_id, text, metadata)
            results.append(VectorSearchResult(chunk=chunk, distance=float(distance)))
        return results

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if not values:
            raise ValueError("embedding vectors cannot be empty")
        return values

    @staticmethod
    def _metadata(chunk: PaperChunk) -> dict[str, str | int | float | bool]:
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "paper_id": chunk.paper_id,
                "arxiv_id": chunk.paper_id,
                "page_numbers": ",".join(str(page) for page in chunk.page_numbers),
                "section": chunk.section or "",
            }
        )
        return metadata

    @staticmethod
    def _chunk_from_result(
        chunk_id: str,
        text: str,
        metadata: Mapping[str, object],
    ) -> PaperChunk:
        paper_id = str(metadata.get("paper_id", metadata.get("arxiv_id", "")))
        raw_pages = str(metadata.get("page_numbers", ""))
        page_numbers = tuple(int(page) for page in raw_pages.split(",") if page)
        raw_section = metadata.get("section")
        section = str(raw_section) if raw_section not in (None, "") else None
        safe_metadata = {
            str(key): value
            for key, value in metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        return PaperChunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            page_numbers=page_numbers,
            section=section,
            metadata=safe_metadata,
        )
