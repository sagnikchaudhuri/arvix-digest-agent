"""Deterministic, page-aware chunking for parsed papers."""

from dataclasses import dataclass
import hashlib
import re

from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.parsing.models import ParsedPaper


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    page_number: int
    section: str | None
    paragraph_number: int


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    page_number: int
    section: str | None
    paragraph_number: int


class TextChunker:
    """Split a parsed paper into stable, overlapping text chunks.

    Chunk sizes and overlap are measured in characters. Boundaries prefer
    paragraph endings and never cross a detected section boundary.
    """

    def __init__(
        self,
        target_chunk_size: int = 1200,
        overlap: int = 180,
        minimum_chunk_size: int | None = None,
    ) -> None:
        if target_chunk_size < 1:
            raise ValueError("target_chunk_size must be positive")
        if overlap < 0 or overlap >= target_chunk_size:
            raise ValueError("overlap must be non-negative and smaller than target_chunk_size")
        if minimum_chunk_size is None:
            minimum_chunk_size = min(250, max(1, target_chunk_size // 4))
        if minimum_chunk_size < 1 or minimum_chunk_size > target_chunk_size:
            raise ValueError("minimum_chunk_size must be between 1 and target_chunk_size")

        self.target_chunk_size = target_chunk_size
        self.overlap = overlap
        self.minimum_chunk_size = minimum_chunk_size

    def chunk(self, paper: ParsedPaper) -> tuple[PaperChunk, ...]:
        paragraphs = self._paragraphs(paper)
        tokens = [
            _Token(word, paragraph.page_number, paragraph.section, paragraph.paragraph_number)
            for paragraph in paragraphs
            for word in paragraph.text.split()
        ]
        if not tokens:
            return ()

        chunks: list[PaperChunk] = []
        start = 0
        while start < len(tokens):
            section = tokens[start].section
            end = start
            current_size = 0
            while end < len(tokens) and tokens[end].section == section:
                separator_size = self._separator_size(tokens, end - 1, end) if end > start else 0
                addition = len(tokens[end].text) + separator_size
                if end > start and current_size + addition > self.target_chunk_size:
                    break
                current_size += addition
                end += 1
                if current_size >= self.target_chunk_size:
                    break

            end = self._prefer_paragraph_boundary(tokens, start, end)
            if end <= start:
                end = start + 1
            text = self._render(tokens, start, end)
            page_numbers = tuple(dict.fromkeys(token.page_number for token in tokens[start:end]))
            index = len(chunks)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{paper.paper.arxiv_id}:{index:04d}:{digest}"
            metadata = {
                "paper_id": paper.paper.arxiv_id,
                "arxiv_id": paper.paper.arxiv_id,
                "title": paper.paper.title,
                "categories": ",".join(paper.paper.categories),
                "page_numbers": ",".join(str(page) for page in page_numbers),
                "section": section or "",
                "chunk_index": index,
            }
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper.arxiv_id,
                    text=text,
                    page_numbers=page_numbers,
                    section=section,
                    metadata=metadata,
                )
            )

            if self.overlap == 0 or end == len(tokens):
                start = end
                continue
            next_start = self._overlap_start(tokens, start, end, section)
            start = next_start if next_start > start else end

        return tuple(chunks)

    def _paragraphs(self, paper: ParsedPaper) -> list[_Paragraph]:
        sections_by_page: dict[int, dict[str, str]] = {}
        for heading in paper.sections:
            sections_by_page.setdefault(heading.page_number, {})[
                self._normalize_heading(heading.title)
            ] = heading.title

        all_headings = sorted(paper.sections, key=lambda heading: heading.page_number)
        paragraphs: list[_Paragraph] = []
        active_section: str | None = None
        paragraph_number = 0

        for page in paper.pages:
            prior_headings = [
                heading for heading in all_headings if heading.page_number < page.page_number
            ]
            if prior_headings:
                active_section = prior_headings[-1].title

            lines: list[str] = []

            def flush() -> None:
                nonlocal paragraph_number
                text = " ".join(" ".join(lines).split())
                if text:
                    paragraphs.append(
                        _Paragraph(text, page.page_number, active_section, paragraph_number)
                    )
                    paragraph_number += 1
                lines.clear()

            heading_map = sections_by_page.get(page.page_number, {})
            for line in page.text.splitlines():
                normalized_heading = self._normalize_heading(line)
                title = heading_map.get(normalized_heading)
                if title is not None:
                    flush()
                    active_section = title
                elif line.strip():
                    lines.append(line.strip())
                else:
                    flush()
            flush()

        return self._merge_tiny_paragraphs(paragraphs)

    def _merge_tiny_paragraphs(self, paragraphs: list[_Paragraph]) -> list[_Paragraph]:
        merged: list[_Paragraph] = []
        for paragraph in paragraphs:
            if (
                merged
                and len(merged[-1].text) < self.minimum_chunk_size
                and merged[-1].section == paragraph.section
                and merged[-1].page_number == paragraph.page_number
            ):
                previous = merged[-1]
                merged[-1] = _Paragraph(
                    f"{previous.text} {paragraph.text}",
                    previous.page_number,
                    paragraph.section,
                    previous.paragraph_number,
                )
            else:
                merged.append(paragraph)
        return merged

    def _prefer_paragraph_boundary(self, tokens: list[_Token], start: int, end: int) -> int:
        if end >= len(tokens):
            return end
        candidates = [
            boundary
            for boundary in range(start + 1, end + 1)
            if tokens[boundary - 1].paragraph_number != tokens[boundary].paragraph_number
            and tokens[boundary].section == tokens[start].section
            and len(self._render(tokens, start, boundary)) >= self.minimum_chunk_size
        ]
        return candidates[-1] if candidates else end

    def _overlap_start(
        self,
        tokens: list[_Token],
        start: int,
        end: int,
        section: str | None,
    ) -> int:
        if tokens[end].section != section:
            return end
        candidate = end
        while candidate > start + 1:
            previous = candidate - 1
            candidate_size = len(self._render(tokens, previous, end))
            if candidate_size > self.overlap:
                break
            candidate = previous
        return candidate

    @staticmethod
    def _separator_size(tokens: list[_Token], previous: int, current: int) -> int:
        return 2 if tokens[previous].paragraph_number != tokens[current].paragraph_number else 1

    @staticmethod
    def _render(tokens: list[_Token], start: int, end: int) -> str:
        rendered: list[str] = []
        for index in range(start, end):
            if index > start:
                rendered.append(
                    "\n\n"
                    if tokens[index - 1].paragraph_number != tokens[index].paragraph_number
                    else " "
                )
            rendered.append(tokens[index].text)
        return "".join(rendered)

    @staticmethod
    def _normalize_heading(value: str) -> str:
        without_number = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", value.strip())
        return re.sub(r"[.:-]+$", "", without_number).strip().casefold()
