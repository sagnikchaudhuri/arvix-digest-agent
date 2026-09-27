"""Local, batched embeddings with lazy sentence-transformer loading."""

from collections.abc import Sequence
from typing import Any, Protocol

from arxiv_digest.config.settings import DEFAULT_EMBEDDING_MODEL
from arxiv_digest.indexing.models import PaperChunk

class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_chunks(
        self, chunks: Sequence[PaperChunk]
    ) -> tuple[tuple[float, ...], ...]: ...


class SentenceTransformerEmbedder:
    """Encode text locally using a configurable Sentence Transformers model."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        if model_name is None:
            from arxiv_digest.config.settings import Settings

            model_name = Settings.from_environment().embedding_model
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = model

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")

        encoded = self._get_model().encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        vectors = tuple(tuple(float(value) for value in row) for row in encoded)
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise RuntimeError("embedding model returned an unexpected output shape")
        if any(len(vector) != len(vectors[0]) for vector in vectors):
            raise RuntimeError("embedding model returned vectors with inconsistent dimensions")
        return vectors

    def embed_chunks(
        self, chunks: Sequence[PaperChunk]
    ) -> tuple[tuple[float, ...], ...]:
        return self.embed_texts([chunk.text for chunk in chunks])

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
