from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from arxiv_digest.indexing import TextChunker
from arxiv_digest.parsing.models import PageText, ParsedPaper, SectionHeading
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.summarization import Evidence, PaperBriefing, PaperSummarizer
from arxiv_digest.summarization.models import NOT_FOUND


def make_metadata() -> PaperMetadata:
    return PaperMetadata(
        title="A Paper About Cache Compression",
        authors=("A. Author", "B. Researcher"),
        arxiv_id="2501.12345v1",
        abstract="Cache compression reduces memory use.",
        published=datetime(2025, 1, 2, tzinfo=timezone.utc),
        categories=("cs.AI",),
        pdf_url="https://arxiv.org/pdf/2501.12345v1",
        abstract_url="https://arxiv.org/abs/2501.12345v1",
    )


def make_parsed(text: str, sections: tuple[SectionHeading, ...] = ()) -> ParsedPaper:
    return ParsedPaper(
        paper=make_metadata(),
        pdf_path=Path("fixture.pdf"),
        full_text=text,
        pages=(PageText(1, text),) if text else (),
        abstract=None,
        abstract_source=None,
        sections=sections,
        references=None,
        warnings=(),
    )


def structured_paper() -> ParsedPaper:
    text = "\n\n".join((
        "Abstract\nWe study the challenge of reducing cache memory while preserving useful information.",
        "Introduction\nThe problem is that large key-value caches consume substantial memory during inference.",
        "Method\nWe propose a grouped quantization method that compresses cached values.",
        "Results\nOur results show reduced memory use with a small change in answer quality.",
        "Limitations\nA limitation is that the method has only been evaluated on two workloads.",
    ))
    headings = tuple(
        SectionHeading(title, 1)
        for title in ("Abstract", "Introduction", "Method", "Results", "Limitations")
    )
    return make_parsed(text, headings)


class FakeSummaryProvider:
    def __init__(self, output: str | None = None, error: Exception | None = None) -> None:
        self.output = output or json.dumps(
            {"summary": "The paper studies cache compression and reports reduced memory use."}
        )
        self.error = error
        self.prompts: list[str] = []
        self.output_limits: list[int] = []
        self.timeouts: list[float | None] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.output

    def generate_with_options(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        timeout_seconds: float | None = None,
    ) -> str:
        self.output_limits.append(max_output_tokens)
        self.timeouts.append(timeout_seconds)
        return self.generate(prompt)


class TestPaperBriefing(unittest.TestCase):
    def test_briefing_works_without_an_available_llm(self) -> None:
        parsed = structured_paper()
        with patch(
            "arxiv_digest.summarization.service.create_llm_provider",
            side_effect=RuntimeError("provider is not configured"),
        ):
            briefing = PaperSummarizer().summarize(make_metadata(), parsed)

        self.assertIsInstance(briefing, PaperBriefing)
        self.assertNotEqual(briefing.plain_english_summary, NOT_FOUND)
        self.assertIn(briefing.problem, parsed.full_text)

    def test_deterministic_extraction_is_stable(self) -> None:
        parsed = structured_paper()
        results = [
            PaperSummarizer(provider=FakeSummaryProvider(error=RuntimeError("offline"))).summarize(
                make_metadata(), parsed
            )
            for _ in range(2)
        ]
        self.assertEqual(results[0], results[1])

    def test_representative_passages_contain_only_valid_chunk_ids(self) -> None:
        parsed = structured_paper()
        chunks = TextChunker().chunk(parsed)
        provider = FakeSummaryProvider()
        PaperSummarizer(provider).summarize(make_metadata(), parsed)

        prompt = provider.prompts[0]
        selected_ids = re.findall(r"CHUNK_ID: ([^\r\n]+)", prompt)
        self.assertTrue(selected_ids)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.assertTrue(set(selected_ids) <= set(chunks_by_id))
        self.assertTrue(all(chunks_by_id[chunk_id].text in prompt for chunk_id in selected_ids))
        self.assertLessEqual(len(prompt), 5500)
        self.assertEqual(provider.output_limits, [300])
        self.assertEqual(provider.timeouts, [90.0])

    def test_evidence_quotes_and_pages_are_authoritative(self) -> None:
        parsed = structured_paper()
        chunks = {chunk.chunk_id: chunk for chunk in TextChunker().chunk(parsed)}
        briefing = PaperSummarizer(FakeSummaryProvider()).summarize(make_metadata(), parsed)

        self.assertTrue(briefing.evidence)
        for evidence in briefing.evidence:
            source = chunks[evidence.chunk_id]
            self.assertEqual(evidence.quote, source.text)
            self.assertEqual(evidence.page_numbers, source.page_numbers)
            self.assertTrue(set(evidence.page_numbers) <= {1})

    def test_unsupported_fields_remain_explicitly_not_found(self) -> None:
        parsed = make_parsed("The paper contains measurements and observations from a collection of samples.")
        briefing = PaperSummarizer(FakeSummaryProvider()).summarize(make_metadata(), parsed)

        self.assertEqual(briefing.problem, NOT_FOUND)
        self.assertEqual(briefing.method, NOT_FOUND)
        self.assertEqual(briefing.key_results, NOT_FOUND)
        self.assertEqual(briefing.limitations, NOT_FOUND)
        self.assertFalse(any(item.field in {"problem", "method", "key_results", "limitations"} for item in briefing.evidence))

    def test_short_paper_generates_a_briefing(self) -> None:
        parsed = make_parsed("The method compresses cached values. Results show lower memory use.")
        briefing = PaperSummarizer(FakeSummaryProvider()).summarize(make_metadata(), parsed)
        self.assertTrue(briefing.plain_english_summary)
        self.assertEqual(len(briefing.suggested_follow_up_questions), 2)

    def test_long_paper_uses_at_most_one_summary_call_and_keeps_all_chunks(self) -> None:
        sentence = "The evaluation reports that the proposed method reduces cache memory use."
        parsed = make_parsed(" ".join(sentence for _ in range(350)))
        all_chunks = TextChunker().chunk(parsed)
        provider = FakeSummaryProvider()
        briefing = PaperSummarizer(provider).summarize(make_metadata(), parsed)

        self.assertGreaterEqual(len(all_chunks), 20)
        self.assertEqual(len(provider.prompts), 1)
        self.assertLessEqual(len(re.findall(r"CHUNK_ID: ([^\r\n]+)", provider.prompts[0])), 5)
        self.assertTrue(briefing.evidence)

    def test_summary_provider_failure_falls_back_without_failing_briefing(self) -> None:
        parsed = structured_paper()
        provider = FakeSummaryProvider(error=TimeoutError("local model timed out"))
        briefing = PaperSummarizer(provider).summarize(make_metadata(), parsed)

        self.assertEqual(len(provider.prompts), 1)
        self.assertNotEqual(briefing.plain_english_summary, NOT_FOUND)
        self.assertIn("cache", briefing.plain_english_summary.casefold())
        self.assertTrue(briefing.evidence)

    def test_malformed_summary_response_falls_back(self) -> None:
        parsed = structured_paper()
        briefing = PaperSummarizer(FakeSummaryProvider(output="not json")).summarize(
            make_metadata(), parsed
        )
        self.assertNotEqual(briefing.plain_english_summary, NOT_FOUND)

    def test_paper_metadata_is_preserved(self) -> None:
        briefing = PaperSummarizer(FakeSummaryProvider()).summarize(
            make_metadata(), structured_paper()
        )
        self.assertEqual(briefing.title, make_metadata().title)
        self.assertEqual(briefing.authors, make_metadata().authors)
        self.assertEqual(briefing.arxiv_id, make_metadata().arxiv_id)
        self.assertEqual(briefing.publication_date, "2025-01-02")
        self.assertEqual(briefing.arxiv_link, make_metadata().abstract_url)

    def test_briefing_model_still_requires_evidence_for_substantive_fields(self) -> None:
        evidence = Evidence("summary", "Source passage.", (1,), "chunk-1")
        briefing = PaperBriefing(
            title=make_metadata().title,
            authors=make_metadata().authors,
            arxiv_id=make_metadata().arxiv_id,
            publication_date="2025-01-02",
            arxiv_link=make_metadata().abstract_url,
            plain_english_summary="An extractive summary.",
            problem=NOT_FOUND,
            method=NOT_FOUND,
            key_results=NOT_FOUND,
            limitations=NOT_FOUND,
            suggested_follow_up_questions=("What should be tested next?",),
            evidence=(evidence,),
        )
        self.assertEqual(briefing.evidence, (evidence,))
        with self.assertRaisesRegex(ValueError, "requires source evidence"):
            PaperBriefing(
                title="Title",
                authors=(),
                arxiv_id="id",
                publication_date="date",
                arxiv_link="url",
                plain_english_summary="Unsupported claim.",
                problem=NOT_FOUND,
                method=NOT_FOUND,
                key_results=NOT_FOUND,
                limitations=NOT_FOUND,
                suggested_follow_up_questions=("What should be tested?",),
                evidence=(),
            )


if __name__ == "__main__":
    unittest.main()
