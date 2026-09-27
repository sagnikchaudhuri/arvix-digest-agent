"""Environment-backed settings shared by the local pipeline components."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PDF_DIR = Path("data/papers")
DEFAULT_VECTOR_STORE_DIR = Path("data/vector_store")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class Settings:
    pdf_dir: Path = DEFAULT_PDF_DIR
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        pdf_dir = os.getenv("PDF_DIR", "").strip() or str(DEFAULT_PDF_DIR)
        vector_store_dir = os.getenv("VECTOR_STORE_DIR", "").strip() or str(
            DEFAULT_VECTOR_STORE_DIR
        )
        return cls(
            pdf_dir=Path(os.path.expandvars(pdf_dir)).expanduser(),
            vector_store_dir=Path(
                os.path.expandvars(vector_store_dir)
            ).expanduser(),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
            or DEFAULT_EMBEDDING_MODEL,
        )
