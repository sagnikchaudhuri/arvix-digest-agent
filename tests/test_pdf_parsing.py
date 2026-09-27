from datetime import datetime, timezone
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from arxiv_digest.parsing import (
    CorruptPDFError,
    InsufficientTextError,
    InvalidPDFError,
    PDFDownloadError,
    PDFHTTPError,
    PDFFetcher,
    PDFParser,
    PDFTextParser,
)
from arxiv_digest.retrieval.models import PaperMetadata


def make_paper() -> PaperMetadata:
    return PaperMetadata(
        title="Test paper",
        authors=("A. Author",),
        arxiv_id="2501.12345v2",
        abstract="Fallback abstract from arXiv metadata.",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        categories=("cs.AI",),
        pdf_url="https://arxiv.org/pdf/2501.12345.pdf",
        abstract_url="https://arxiv.org/abs/2501.12345v2",
    )


def make_pdf(page_lines: list[list[str]]) -> bytes:
    """Build a tiny text PDF using pypdf's writer primitives."""
    writer = PdfWriter()
    for lines in page_lines:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = writer._add_object(font)
        font_resources = DictionaryObject({NameObject("/F1"): font_ref})
        resources = DictionaryObject({NameObject("/Font"): font_resources})
        page[NameObject("/Resources")] = resources

        operations = []
        for line_number, line in enumerate(lines):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            y_position = 750 - 18 * line_number
            operations.append(
                f"BT /F1 12 Tf 50 {y_position} Td ({escaped}) Tj ET"
            )
        content = DecodedStreamObject()
        content.set_data("\n".join(operations).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class TestPDFFetcher(unittest.TestCase):
    def test_downloads_and_reuses_cached_pdf(self) -> None:
        payload = make_pdf([["A valid small PDF fixture with extractable text."]])
        downloader = Mock(return_value=payload)
        with TemporaryDirectory() as temporary_directory:
            fetcher = PDFFetcher(temporary_directory, downloader=downloader)
            first_path = fetcher.fetch(make_paper())
            second_path = fetcher.fetch(make_paper())

            self.assertEqual(first_path, second_path)
            self.assertTrue(first_path.is_file())
            self.assertEqual(first_path.read_bytes(), payload)
            downloader.assert_called_once_with("https://arxiv.org/pdf/2501.12345.pdf")

    def test_network_failure_is_structured(self) -> None:
        def fail(_url: str) -> bytes:
            raise URLError("offline")

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(PDFDownloadError) as context:
                PDFFetcher(temporary_directory, downloader=fail).fetch(make_paper())
        self.assertEqual(context.exception.code, "download_failed")

    def test_http_failure_preserves_status_code(self) -> None:
        def fail(_url: str) -> bytes:
            raise HTTPError("https://arxiv.org/pdf/2501.12345.pdf", 503, "unavailable", None, None)

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(PDFHTTPError) as context:
                PDFFetcher(temporary_directory, downloader=fail).fetch(make_paper())
        self.assertEqual(context.exception.status_code, 503)

    def test_non_pdf_response_is_rejected_and_not_cached(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fetcher = PDFFetcher(temporary_directory, downloader=lambda _url: b"<html>error</html>")
            with self.assertRaises(InvalidPDFError):
                fetcher.fetch(make_paper())
            self.assertFalse(Path(temporary_directory, "2501.12345v2.pdf").exists())

    def test_corrupt_pdf_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            fetcher = PDFFetcher(
                temporary_directory,
                downloader=lambda _url: b"%PDF-1.7\nthis is not a valid PDF",
            )
            with self.assertRaises(CorruptPDFError):
                fetcher.fetch(make_paper())

    def test_only_official_arxiv_pdf_urls_are_accepted(self) -> None:
        paper = make_paper()
        paper = replace(paper, pdf_url="https://example.com/pdf/2501.12345.pdf")
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaises(PDFDownloadError):
                PDFFetcher(temporary_directory, downloader=lambda _url: b"").fetch(paper)


class TestPDFTextParser(unittest.TestCase):
    def test_extracts_page_text_and_warns_when_a_page_fails(self) -> None:
        class GoodPage:
            def extract_text(self) -> str:
                return "Extractable text " * 20

        class BadPage:
            def extract_text(self) -> str:
                raise ValueError("bad page")

        reader = type("Reader", (), {"pages": [GoodPage(), BadPage()]})()
        with patch("arxiv_digest.parsing.pdf.PdfReader", return_value=reader):
            parsed = PDFTextParser(minimum_text_characters=20).parse(
                make_paper(), "unused.pdf"
            )

        self.assertEqual([page.page_number for page in parsed.pages], [1, 2])
        self.assertEqual(parsed.pages[1].text, "")
        self.assertEqual(parsed.warnings[0].code, "page_extraction_failed")
        self.assertEqual(parsed.warnings[0].page_number, 2)

    def test_insufficient_text_explains_ocr_is_not_supported(self) -> None:
        payload = make_pdf([["Only a few words."]])
        with TemporaryDirectory() as temporary_directory:
            parser = PDFParser(
                temporary_directory,
                downloader=lambda _url: payload,
                minimum_text_characters=100,
            )
            with self.assertRaises(InsufficientTextError) as context:
                parser.parse(make_paper())

        self.assertEqual(context.exception.code, "insufficient_text")
        self.assertTrue(any(warning.code == "ocr_unsupported" for warning in context.exception.warnings))
        self.assertIn("OCR is not performed", context.exception.warnings[-1].message)

    def test_generated_pdf_integration_extracts_structure_and_pages(self) -> None:
        fixture = make_pdf(
            [
                [
                    "Abstract",
                    "This paper evaluates a small test approach and reports measurable improvements across several useful benchmark tasks.",
                    "The results show reliable gains while keeping computation and memory requirements modest for practical applications.",
                    "1 Introduction",
                    "This introduction contains enough ordinary selectable text to verify extraction from the generated PDF fixture.",
                ],
                [
                    "2 Method",
                    "We compare a baseline approach with a straightforward alternative across multiple representative evaluation settings.",
                    "References",
                    "[1] A. Author. A reference paper. 2024.",
                ],
            ]
        )
        with TemporaryDirectory() as temporary_directory:
            parsed = PDFParser(
                temporary_directory,
                downloader=lambda _url: fixture,
            ).parse(make_paper())
            self.assertTrue(parsed.pdf_path.is_file())

        self.assertEqual(len(parsed.pages), 2)
        self.assertEqual([page.page_number for page in parsed.pages], [1, 2])
        self.assertIn("This paper evaluates", parsed.full_text)
        self.assertTrue(parsed.abstract.startswith("This paper evaluates"))
        self.assertEqual(parsed.abstract_source, "pdf")
        self.assertIn("Introduction", [section.title for section in parsed.sections])
        self.assertIn("Method", [section.title for section in parsed.sections])
        self.assertEqual(parsed.sections[-1].page_number, 2)
        self.assertIn("[1] A. Author", parsed.references)


if __name__ == "__main__":
    unittest.main()
