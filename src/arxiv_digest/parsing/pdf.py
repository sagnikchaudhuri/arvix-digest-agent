"""Fetch arXiv PDFs into a local cache and extract text with pypdf."""

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

from arxiv_digest.config.settings import DEFAULT_PDF_DIR
from arxiv_digest.parsing.errors import (
    CorruptPDFError,
    InsufficientTextError,
    InvalidPDFError,
    PDFDownloadError,
    PDFHTTPError,
    PDFProcessingError,
)
from arxiv_digest.parsing.models import (
    PageText,
    ParseWarning,
    ParsedPaper,
    SectionHeading,
)
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.retrieval.query import QueryKind, classify_query, parse_arxiv_url


DEFAULT_CACHE_DIR = DEFAULT_PDF_DIR
DEFAULT_MINIMUM_TEXT_CHARACTERS = 100
_OFFICIAL_ARXIV_HOSTS = {"arxiv.org", "export.arxiv.org"}
_USER_AGENT = "arxiv-digest-agent/0.1.0"


def _official_pdf_url(url: str, arxiv_id: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise PDFDownloadError(arxiv_id, "Invalid PDF URL.") from exc

    url_id = parse_arxiv_url(url)
    unversioned_id = re.sub(r"v\d+$", "", arxiv_id)
    if (
        parsed.scheme != "https"
        or host not in _OFFICIAL_ARXIV_HOSTS
        or port not in (None, 443)
        or not parsed.path.startswith("/pdf/")
        or url_id not in {arxiv_id, unversioned_id}
    ):
        raise PDFDownloadError(
            arxiv_id, "PDF URL must be the official arXiv PDF URL for this paper."
        )
    return url


def _validate_pdf_bytes(data: bytes, arxiv_id: str) -> None:
    if not data.startswith(b"%PDF-"):
        raise InvalidPDFError(arxiv_id, "The download is not a PDF file.")
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if len(reader.pages) == 0:
            raise CorruptPDFError(arxiv_id, "The PDF contains no readable pages.")
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise CorruptPDFError(arxiv_id, "The PDF file is corrupted or unreadable.") from exc


class PDFFetcher:
    """Download and cache validated PDFs from official arXiv URLs."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        downloader: Callable[[str], bytes] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if cache_dir is None:
            from arxiv_digest.config.settings import Settings

            cache_dir = Settings.from_environment().pdf_dir
        self.cache_dir = Path(cache_dir)
        self._downloader = downloader or self._download
        self._timeout_seconds = timeout_seconds

    def fetch(self, paper: PaperMetadata) -> Path:
        """Return a cached local PDF, downloading it only when absent."""
        intent = classify_query(paper.arxiv_id)
        if intent.kind is not QueryKind.PAPER_ID:
            raise PDFDownloadError(paper.arxiv_id, "Paper metadata has an invalid arXiv ID.")
        url = _official_pdf_url(paper.pdf_url, paper.arxiv_id)
        filename = f"{paper.arxiv_id.replace('/', '_')}.pdf"
        target = self.cache_dir / filename

        if target.is_file():
            return target

        try:
            payload = self._downloader(url)
        except HTTPError as exc:
            raise PDFHTTPError(paper.arxiv_id, exc.code, "arXiv returned an HTTP error.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PDFDownloadError(paper.arxiv_id, f"PDF download failed: {exc}") from exc
        except Exception as exc:
            raise PDFDownloadError(paper.arxiv_id, f"PDF download failed: {exc}") from exc

        if not isinstance(payload, bytes):
            raise InvalidPDFError(paper.arxiv_id, "The downloader did not return PDF bytes.")
        _validate_pdf_bytes(payload, paper.arxiv_id)

        temporary_path: Path | None = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.cache_dir,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            temporary_path.replace(target)
        except OSError as exc:
            raise PDFDownloadError(
                paper.arxiv_id, f"Could not write the PDF cache: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return target

    def _download(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(request, timeout=self._timeout_seconds) as response:
            final_host = urlparse(response.geturl()).hostname
            if final_host not in _OFFICIAL_ARXIV_HOSTS:
                raise URLError("PDF redirect left the official arXiv domain.")
            return response.read()


_KNOWN_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "bibliography",
    "conclusion",
    "conclusions",
    "discussion",
    "experiments",
    "experimental results",
    "introduction",
    "limitations",
    "method",
    "methodology",
    "preliminaries",
    "related work",
    "references",
    "results",
}


def _heading_title(line: str) -> str | None:
    candidate = line.strip()
    if not candidate or len(candidate) > 110:
        return None
    numbered = re.fullmatch(r"\d+(?:\.\d+)*\.?\s+(.+)", candidate)
    if numbered:
        title = numbered.group(1).strip().rstrip(" .:")
        return title or None
    normalized = re.sub(r"[.:-]+$", "", candidate).strip().casefold()
    if normalized in _KNOWN_HEADINGS:
        return candidate.rstrip(" .:")
    return None


def _paper_sections(pages: tuple[PageText, ...]) -> tuple[SectionHeading, ...]:
    sections: list[SectionHeading] = []
    seen: set[tuple[int, str]] = set()
    for page in pages:
        for line in page.text.splitlines():
            title = _heading_title(line)
            key = (page.page_number, title.casefold()) if title else None
            if title and key not in seen:
                sections.append(SectionHeading(title, page.page_number))
                seen.add(key)
    return tuple(sections)


def _abstract_from_pages(pages: tuple[PageText, ...]) -> str | None:
    lines = [line.strip() for page in pages for line in page.text.splitlines()]
    start: int | None = None
    abstract_parts: list[str] = []
    inline_abstract = re.compile(
        r"^abstract\b\s*(?:[:\-\u2013\u2014]\s*)?(.*)$", re.IGNORECASE
    )
    for index, line in enumerate(lines):
        match = inline_abstract.match(line)
        if match:
            start = index + 1
            if match.group(1).strip():
                abstract_parts.append(match.group(1).strip())
            break
    if start is None:
        return None

    for line in lines[start:]:
        heading = _heading_title(line)
        if heading and heading.casefold() != "abstract":
            break
        if line:
            abstract_parts.append(line)
    text = " ".join(abstract_parts).strip()
    return text or None


def _references_from_pages(pages: tuple[PageText, ...]) -> str | None:
    started = False
    reference_lines: list[str] = []
    for page in pages:
        for line in page.text.splitlines():
            heading = _heading_title(line)
            if not started and heading and heading.casefold() in {"references", "bibliography"}:
                started = True
                continue
            if started:
                reference_lines.append(line)
    references = "\n".join(reference_lines).strip()
    return references or None


class PDFTextParser:
    """Extract page text and best-effort paper structure with pypdf."""

    def __init__(self, minimum_text_characters: int = DEFAULT_MINIMUM_TEXT_CHARACTERS) -> None:
        if minimum_text_characters < 1:
            raise ValueError("minimum_text_characters must be positive")
        self.minimum_text_characters = minimum_text_characters

    def parse(self, paper: PaperMetadata, pdf_path: str | Path) -> ParsedPaper:
        try:
            reader = PdfReader(str(pdf_path), strict=False)
            pdf_pages = reader.pages
            if len(pdf_pages) == 0:
                raise CorruptPDFError(paper.arxiv_id, "The PDF contains no readable pages.")
        except PDFProcessingError:
            raise
        except Exception as exc:
            raise CorruptPDFError(paper.arxiv_id, "The PDF file is corrupted or unreadable.") from exc

        pages: list[PageText] = []
        warnings: list[ParseWarning] = []
        for page_number, page in enumerate(pdf_pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                text = ""
                warnings.append(
                    ParseWarning(
                        code="page_extraction_failed",
                        message=f"Page text could not be extracted ({type(exc).__name__}).",
                        page_number=page_number,
                    )
                )
            pages.append(PageText(page_number=page_number, text=text.strip()))

        page_tuple = tuple(pages)
        full_text = "\n\n".join(page.text for page in page_tuple if page.text)
        character_count = len(re.sub(r"\s", "", full_text))
        if character_count < self.minimum_text_characters:
            warnings.append(
                ParseWarning(
                    code="ocr_unsupported",
                    message=(
                        "The PDF has insufficient extractable text. It may be scanned or image-only; "
                        "OCR is not performed."
                    ),
                )
            )
            raise InsufficientTextError(
                paper.arxiv_id,
                f"Only {character_count} non-whitespace characters could be extracted.",
                warnings=tuple(warnings),
            )

        sections = _paper_sections(page_tuple)
        abstract = _abstract_from_pages(page_tuple)
        abstract_source = "pdf"
        if abstract is None and paper.abstract.strip():
            abstract = paper.abstract.strip()
            abstract_source = "metadata"
        elif abstract is None:
            abstract_source = None
            warnings.append(
                ParseWarning(
                    code="abstract_not_found",
                    message="No abstract was found in the PDF or metadata.",
                )
            )

        return ParsedPaper(
            paper=paper,
            pdf_path=Path(pdf_path),
            full_text=full_text,
            pages=page_tuple,
            abstract=abstract,
            abstract_source=abstract_source,
            sections=sections,
            references=_references_from_pages(page_tuple),
            warnings=tuple(warnings),
        )


class PDFParser:
    """Fetch a paper PDF and return normalized, page-aware parsed content."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        downloader: Callable[[str], bytes] | None = None,
        minimum_text_characters: int = DEFAULT_MINIMUM_TEXT_CHARACTERS,
    ) -> None:
        self.fetcher = PDFFetcher(cache_dir=cache_dir, downloader=downloader)
        self.text_parser = PDFTextParser(minimum_text_characters=minimum_text_characters)

    def parse(self, paper: PaperMetadata) -> ParsedPaper:
        pdf_path = self.fetcher.fetch(paper)
        return self.text_parser.parse(paper, pdf_path)
