from datetime import datetime, timezone
from pathlib import Path
import unittest

from arxiv_digest.indexing import PaperChunk, SentenceTransformerEmbedder, TextChunker
from arxiv_digest.parsing.models import PageText, ParsedPaper, SectionHeading
from arxiv_digest.retrieval.models import PaperMetadata


def make_parsed_paper(pages: list[str], headings: list[SectionHeading]) -> ParsedPaper:
    metadata = PaperMetadata(
        title="Chunking test paper",
        authors=("A. Author",),
        arxiv_id="2501.12345v2",
        abstract="A short test abstract.",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        categories=("cs.CL",),
        pdf_url="https://arxiv.org/pdf/2501.12345.pdf",
        abstract_url="https://arxiv.org/abs/2501.12345v2",
    )
    page_texts = tuple(PageText(index, text) for index, text in enumerate(pages, start=1))
    return ParsedPaper(
        paper=metadata,
        pdf_path=Path("fixture.pdf"),
        full_text="\n\n".join(pages),
        pages=page_texts,
        abstract=metadata.abstract,
        abstract_source="metadata",
        sections=tuple(headings),
        references=None,
        warnings=(),
    )


class FakeEncoder:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.arguments = kwargs
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


class TestTextChunker(unittest.TestCase):
    def test_chunking_is_deterministic(self) -> None:
        text = " ".join(f"term{index:02d}" for index in range(45))
        paper = make_parsed_paper(
            [f"Introduction\n\n{text}"],
            [SectionHeading("Introduction", 1)],
        )
        chunker = TextChunker(target_chunk_size=80, overlap=12, minimum_chunk_size=20)

        first = chunker.chunk(paper)
        second = chunker.chunk(paper)

        self.assertEqual(first, second)
        self.assertTrue(all(chunk.chunk_id == other.chunk_id for chunk, other in zip(first, second)))

    def test_adjacent_chunks_have_configured_word_safe_overlap(self) -> None:
        text = " ".join(f"token{index:02d}" for index in range(40))
        paper = make_parsed_paper(
            [f"Introduction\n{text}"],
            [SectionHeading("Introduction", 1)],
        )
        chunks = TextChunker(
            target_chunk_size=55,
            overlap=14,
            minimum_chunk_size=10,
        ).chunk(paper)

        self.assertGreater(len(chunks), 1)
        left = chunks[0].text.split()
        right = chunks[1].text.split()
        self.assertTrue(any(left[-size:] == right[:size] for size in range(1, 4)))

    def test_page_provenance_includes_all_source_pages(self) -> None:
        paper = make_parsed_paper(
            [
                "Introduction\nFirst page contributes useful semantic content for retrieval.",
                "Second page continues the same section with additional meaningful content.",
            ],
            [SectionHeading("Introduction", 1)],
        )
        chunks = TextChunker(target_chunk_size=500, overlap=0).chunk(paper)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_numbers, (1, 2))
        self.assertEqual(chunks[0].paper_id, "2501.12345v2")
        self.assertEqual(chunks[0].metadata["page_numbers"], "1,2")

    def test_section_metadata_is_preserved_and_forms_a_boundary(self) -> None:
        paper = make_parsed_paper(
            [
                "Introduction\nThe introduction text describes the problem and motivation.\n\n"
                "Methods\nThe methods text explains the proposed evaluation procedure."
            ],
            [SectionHeading("Introduction", 1), SectionHeading("Methods", 1)],
        )
        chunks = TextChunker(target_chunk_size=500, overlap=0).chunk(paper)

        self.assertEqual([chunk.section for chunk in chunks], ["Introduction", "Methods"])
        self.assertEqual(chunks[1].metadata["section"], "Methods")


class TestSentenceTransformerEmbedder(unittest.TestCase):
    def test_embedding_batch_has_expected_shape_without_loading_model(self) -> None:
        chunks = (
            PaperChunk("c1", "p1", "first text", (1,), "Intro", {}),
            PaperChunk("c2", "p1", "second text", (2,), "Results", {}),
        )
        fake_model = FakeEncoder()
        embedder = SentenceTransformerEmbedder(
            model_name="local-test-model", batch_size=2, model=fake_model
        )

        vectors = embedder.embed_chunks(chunks)

        self.assertEqual(vectors, ((1.0, 10.0), (2.0, 11.0)))
        self.assertEqual(len(vectors), len(chunks))
        self.assertEqual({len(vector) for vector in vectors}, {2})
        self.assertEqual(fake_model.arguments["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
