from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from arxiv_digest.retrieval.arxiv_client import ArxivRetriever, paper_metadata_from_result
from arxiv_digest.retrieval.errors import (
    ArxivApiError,
    InvalidQueryError,
    NoResultsError,
    UnexpectedArxivResponseError,
)
from arxiv_digest.retrieval.query import QueryKind, classify_query, parse_arxiv_url


def make_result() -> SimpleNamespace:
    return SimpleNamespace(
        title="KV Cache Compression",
        authors=[SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Alan Turing")],
        summary="A paper abstract.",
        published=datetime(2025, 1, 15, tzinfo=timezone.utc),
        categories=["cs.CL", "cs.LG"],
        pdf_url="https://arxiv.org/pdf/2501.01234.pdf",
        entry_id="http://arxiv.org/abs/2501.01234v2",
    )


class SearchSpec:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class FakeClient:
    def __init__(self, results: list[object] | Exception) -> None:
        self._results = results
        self.searches: list[SearchSpec] = []

    def results(self, search: SearchSpec) -> list[object]:
        self.searches.append(search)
        if isinstance(self._results, Exception):
            raise self._results
        return self._results


class TestQueryClassification(unittest.TestCase):
    def test_parses_valid_arxiv_id(self) -> None:
        result = classify_query("2301.12345v2")
        self.assertEqual(result.kind, QueryKind.PAPER_ID)
        self.assertEqual(result.arxiv_id, "2301.12345v2")

    def test_parses_arxiv_url(self) -> None:
        url = "https://arxiv.org/abs/2301.12345v2"
        self.assertEqual(parse_arxiv_url(url), "2301.12345v2")
        self.assertEqual(classify_query(url).kind, QueryKind.PAPER_URL)
        self.assertEqual(
            parse_arxiv_url("https://arxiv.org/pdf/hep-th/9901001v1.pdf"),
            "hep-th/9901001v1",
        )

    def test_classifies_topic(self) -> None:
        result = classify_query("recent work on KV-cache compression for LLMs")
        self.assertEqual(result.kind, QueryKind.TOPIC)
        self.assertEqual(result.query, "recent work on KV-cache compression for LLMs")

    def test_invalid_input(self) -> None:
        self.assertEqual(classify_query("  ").kind, QueryKind.INVALID)
        self.assertEqual(classify_query("2301.12").kind, QueryKind.INVALID)
        self.assertEqual(classify_query("2300.12345").kind, QueryKind.INVALID)
        self.assertEqual(classify_query("2301.12345v0").kind, QueryKind.INVALID)
        self.assertEqual(classify_query("https://example.com/abs/2301.12345").kind, QueryKind.INVALID)
        with self.assertRaises(InvalidQueryError):
            ArxivRetriever(client=FakeClient([]), search_factory=SearchSpec).retrieve("")


class TestArxivRetrieval(unittest.TestCase):
    def test_converts_api_result_to_paper_metadata(self) -> None:
        metadata = paper_metadata_from_result(make_result())
        self.assertEqual(metadata.title, "KV Cache Compression")
        self.assertEqual(metadata.authors, ("Ada Lovelace", "Alan Turing"))
        self.assertEqual(metadata.arxiv_id, "2501.01234v2")
        self.assertEqual(metadata.abstract, "A paper abstract.")
        self.assertEqual(metadata.published.year, 2025)
        self.assertEqual(metadata.categories, ("cs.CL", "cs.LG"))
        self.assertEqual(metadata.pdf_url, "https://arxiv.org/pdf/2501.01234.pdf")
        self.assertEqual(metadata.abstract_url, "https://arxiv.org/abs/2501.01234v2")

    def test_specific_id_uses_id_list_lookup(self) -> None:
        client = FakeClient([make_result()])
        papers = ArxivRetriever(client, SearchSpec).retrieve("2501.01234")
        self.assertEqual(len(papers), 1)
        self.assertEqual(client.searches[0].kwargs, {"id_list": ["2501.01234"], "max_results": 1})

    def test_topic_uses_api_search(self) -> None:
        client = FakeClient([make_result()])
        ArxivRetriever(client, SearchSpec, max_results=4).retrieve("KV cache compression")
        self.assertEqual(
            client.searches[0].kwargs,
            {"query": "KV cache compression", "max_results": 4},
        )

    def test_empty_results_raise_no_results_error(self) -> None:
        with self.assertRaises(NoResultsError):
            ArxivRetriever(FakeClient([]), SearchSpec).retrieve("quantum computing")

    def test_api_failure_is_reported(self) -> None:
        with self.assertRaises(ArxivApiError):
            ArxivRetriever(FakeClient(RuntimeError("offline")), SearchSpec).retrieve("quantum")

    def test_unexpected_result_is_reported(self) -> None:
        with self.assertRaises(UnexpectedArxivResponseError):
            paper_metadata_from_result(SimpleNamespace(title="missing fields"))


if __name__ == "__main__":
    unittest.main()
