import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from arxiv_digest.config.settings import Settings
from arxiv_digest.indexing.embeddings import DEFAULT_EMBEDDING_MODEL
from arxiv_digest.indexing.embeddings import SentenceTransformerEmbedder
from arxiv_digest.parsing.pdf import PDFParser
from arxiv_digest.parsing.pdf import DEFAULT_CACHE_DIR
from arxiv_digest.vector_store.chroma import ChromaVectorStore
from arxiv_digest.vector_store.chroma import DEFAULT_VECTOR_STORE_DIR


class SettingsTests(unittest.TestCase):
    def test_environment_overrides_local_data_and_embedding_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PDF_DIR": "custom/papers",
                "VECTOR_STORE_DIR": "custom/chroma",
                "EMBEDDING_MODEL": "local/test-model",
            },
        ):
            settings = Settings.from_environment()

        self.assertEqual(settings.pdf_dir, Path("custom/papers"))
        self.assertEqual(settings.vector_store_dir, Path("custom/chroma"))
        self.assertEqual(settings.embedding_model, "local/test-model")

    def test_defaults_match_component_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {"PDF_DIR": "", "VECTOR_STORE_DIR": "", "EMBEDDING_MODEL": ""},
        ):
            settings = Settings.from_environment()

        self.assertEqual(settings.pdf_dir, DEFAULT_CACHE_DIR)
        self.assertEqual(settings.vector_store_dir, DEFAULT_VECTOR_STORE_DIR)
        self.assertEqual(settings.embedding_model, DEFAULT_EMBEDDING_MODEL)

    def test_default_components_use_environment_overrides(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "PDF_DIR": str(root / "papers"),
                    "VECTOR_STORE_DIR": str(root / "vectors"),
                    "EMBEDDING_MODEL": "local/configured-model",
                },
            ):
                parser = PDFParser()
                embedder = SentenceTransformerEmbedder()
                store = ChromaVectorStore(collection_name="settings_test")
                try:
                    self.assertEqual(parser.fetcher.cache_dir, root / "papers")
                    self.assertEqual(store.persist_directory, root / "vectors")
                    self.assertEqual(embedder.model_name, "local/configured-model")
                finally:
                    store.close()


if __name__ == "__main__":
    unittest.main()
