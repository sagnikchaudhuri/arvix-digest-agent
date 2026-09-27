from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from arxiv_digest.indexing import PaperChunk, TextChunker
from arxiv_digest.indexing.embeddings import SentenceTransformerEmbedder
from arxiv_digest.parsing.models import PageText, ParsedPaper, SectionHeading
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.vector_store import ChromaVectorStore


def make_chunk(chunk_id: str, paper_id: str, text: str, page: int, section: str) -> PaperChunk:
    return PaperChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=text,
        page_numbers=(page,),
        section=section,
        metadata={
            "paper_id": paper_id,
            "arxiv_id": paper_id,
            "title": f"Paper {paper_id}",
            "categories": "cs.AI",
            "page_numbers": str(page),
            "section": section,
        },
    )


def make_parsed_paper() -> ParsedPaper:
    metadata = PaperMetadata(
        title="Vector integration paper",
        authors=("A. Author",),
        arxiv_id="2501.12345v2",
        abstract="A deterministic vector integration fixture.",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        categories=("cs.AI",),
        pdf_url="https://arxiv.org/pdf/2501.12345.pdf",
        abstract_url="https://arxiv.org/abs/2501.12345v2",
    )
    pages = (
        PageText(
            1,
            "Introduction\nKV cache compression reduces memory use in long context generation.",
        ),
        PageText(
            2,
            "Results\nAttention accuracy remains stable across the evaluated language models.",
        ),
    )
    return ParsedPaper(
        paper=metadata,
        pdf_path=Path("fixture.pdf"),
        full_text="\n\n".join(page.text for page in pages),
        pages=pages,
        abstract=metadata.abstract,
        abstract_source="metadata",
        sections=(SectionHeading("Introduction", 1), SectionHeading("Results", 2)),
        references=None,
        warnings=(),
    )


class DeterministicFakeModel:
    def encode(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [float("cache" in normalized), float("attention" in normalized), 0.1]
            )
        return vectors


class TestChromaVectorStore(unittest.TestCase):
    def test_upsert_and_search_return_original_chunk_data(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with ChromaVectorStore(temporary_directory, "roundtrip") as store:
                expected = make_chunk("chunk-a", "paper-a", "cache compression text", 3, "Methods")
                store.upsert([expected], [[1.0, 0.0]])
                results = store.search([1.0, 0.0], top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, expected.chunk_id)
        self.assertEqual(results[0].chunk.text, expected.text)
        self.assertEqual(results[0].chunk.page_numbers, (3,))
        self.assertEqual(results[0].chunk.section, "Methods")
        self.assertEqual(results[0].chunk.paper_id, "paper-a")

    def test_search_filters_by_paper_id(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with ChromaVectorStore(temporary_directory, "paper_filter") as store:
                store.upsert(
                    [
                        make_chunk("paper-a-1", "paper-a", "paper a content", 1, "Intro"),
                        make_chunk("paper-b-1", "paper-b", "paper b content", 2, "Intro"),
                    ],
                    [[1.0, 0.0], [0.0, 1.0]],
                )
                results = store.search([0.0, 1.0], top_k=2, paper_id="paper-a")

        self.assertEqual([result.chunk.paper_id for result in results], ["paper-a"])

    def test_upsert_replaces_stale_chunks_for_reindexed_paper_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with ChromaVectorStore(temporary_directory, "replace_stale") as store:
                store.upsert(
                    [
                        make_chunk("old-a-1", "paper-a", "old intro", 1, "Introduction"),
                        make_chunk("old-a-2", "paper-a", "old result", 2, "Results"),
                        make_chunk("paper-b-1", "paper-b", "other paper remains", 3, "Intro"),
                    ],
                    [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
                )
                current = [
                    make_chunk("new-a-1", "paper-a", "current intro", 1, "Introduction"),
                    make_chunk("new-a-2", "paper-a", "current results", 2, "Results"),
                ]
                store.upsert(current, [[1.0, 0.0], [0.0, 1.0]])
                paper_a = store.search([1.0, 0.0], top_k=10, paper_id="paper-a")
                paper_b = store.search([1.0, 0.0], top_k=10, paper_id="paper-b")

        self.assertEqual({item.chunk.chunk_id for item in paper_a}, {"new-a-1", "new-a-2"})
        self.assertEqual([item.chunk.chunk_id for item in paper_b], ["paper-b-1"])

    def test_chunk_to_fake_embedding_to_chroma_integration(self) -> None:
        chunks = TextChunker(target_chunk_size=500, overlap=0).chunk(make_parsed_paper())
        embedder = SentenceTransformerEmbedder(
            model_name="unused",
            batch_size=8,
            model=DeterministicFakeModel(),
        )
        embeddings = embedder.embed_chunks(chunks)
        query_vector = embedder.embed_texts(["cache compression"])[0]

        with TemporaryDirectory() as temporary_directory:
            with ChromaVectorStore(temporary_directory, "pipeline") as store:
                store.upsert(chunks, embeddings)
                results = store.search(query_vector, top_k=2, paper_id="2501.12345v2")

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].chunk.section, "Introduction")
        self.assertIn("cache compression", results[0].chunk.text.casefold())
        self.assertIn(1, results[0].chunk.page_numbers)


if __name__ == "__main__":
    unittest.main()
