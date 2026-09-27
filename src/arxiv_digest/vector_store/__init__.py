"""Local vector storage with a backend-neutral interface."""

from arxiv_digest.vector_store.chroma import ChromaVectorStore
from arxiv_digest.vector_store.store import VectorSearchResult, VectorStore

__all__ = ["ChromaVectorStore", "VectorSearchResult", "VectorStore"]
