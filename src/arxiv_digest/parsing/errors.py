"""Structured failures raised while downloading or parsing PDFs."""

from arxiv_digest.parsing.models import ParseWarning


class PDFProcessingError(Exception):
    def __init__(
        self,
        arxiv_id: str,
        message: str,
        *,
        code: str,
        warnings: tuple[ParseWarning, ...] = (),
    ) -> None:
        super().__init__(message)
        self.arxiv_id = arxiv_id
        self.message = message
        self.code = code
        self.warnings = warnings


class PDFDownloadError(PDFProcessingError):
    def __init__(self, arxiv_id: str, message: str) -> None:
        super().__init__(arxiv_id, message, code="download_failed")


class PDFHTTPError(PDFProcessingError):
    def __init__(self, arxiv_id: str, status_code: int, message: str) -> None:
        super().__init__(arxiv_id, message, code="http_error")
        self.status_code = status_code


class InvalidPDFError(PDFProcessingError):
    def __init__(self, arxiv_id: str, message: str) -> None:
        super().__init__(arxiv_id, message, code="invalid_pdf")


class CorruptPDFError(PDFProcessingError):
    def __init__(self, arxiv_id: str, message: str) -> None:
        super().__init__(arxiv_id, message, code="corrupt_pdf")


class InsufficientTextError(PDFProcessingError):
    def __init__(self, arxiv_id: str, message: str, *, warnings: tuple[ParseWarning, ...]) -> None:
        super().__init__(arxiv_id, message, code="insufficient_text", warnings=warnings)
