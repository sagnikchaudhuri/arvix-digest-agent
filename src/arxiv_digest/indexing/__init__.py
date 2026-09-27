"""Page-aware text chunking and local embedding generation."""

from arxiv_digest.indexing.chunking import TextChunker
from arxiv_digest.indexing.embeddings import Embedder, SentenceTransformerEmbedder
from arxiv_digest.indexing.models import PaperChunk

__all__ = ["Embedder", "PaperChunk", "SentenceTransformerEmbedder", "TextChunker"]
