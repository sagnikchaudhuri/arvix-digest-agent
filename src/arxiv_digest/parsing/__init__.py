"""PDF fetching and paper text parsing."""

from arxiv_digest.parsing.errors import (
    CorruptPDFError,
    InsufficientTextError,
    InvalidPDFError,
    PDFDownloadError,
    PDFHTTPError,
    PDFProcessingError,
)
from arxiv_digest.parsing.models import PageText, ParseWarning, ParsedPaper, SectionHeading
from arxiv_digest.parsing.pdf import PDFParser, PDFFetcher, PDFTextParser

__all__ = [
    "CorruptPDFError",
    "InsufficientTextError",
    "InvalidPDFError",
    "PageText",
    "PDFDownloadError",
    "PDFHTTPError",
    "PDFParser",
    "PDFProcessingError",
    "PDFFetcher",
    "PDFTextParser",
    "ParseWarning",
    "ParsedPaper",
    "SectionHeading",
]
