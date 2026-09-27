"""arXiv API access, kept independent from agent orchestration."""

from collections.abc import Callable
from datetime import datetime

from arxiv_digest.retrieval.errors import (
    ArxivApiError,
    InvalidQueryError,
    NoResultsError,
    UnexpectedArxivResponseError,
)
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.retrieval.query import QueryKind, classify_query


def paper_metadata_from_result(result: object) -> PaperMetadata:
    """Convert a result from the arxiv package into the application model."""
    try:
        title = result.title
        raw_authors = result.authors
        raw_categories = result.categories
        if isinstance(raw_authors, (str, bytes)) or isinstance(raw_categories, (str, bytes)):
            raise TypeError("author and category collections must be sequences")
        authors = tuple(
            author if isinstance(author, str) else author.name
            for author in raw_authors
        )
        abstract = result.summary
        published = result.published
        categories = tuple(raw_categories)
        pdf_url = result.pdf_url
        entry_id = result.entry_id
    except (AttributeError, TypeError) as exc:
        raise UnexpectedArxivResponseError(
            "arXiv returned a result with missing or invalid metadata."
        ) from exc

    if not isinstance(title, str) or not title.strip():
        raise UnexpectedArxivResponseError("arXiv result has no valid title.")
    if not authors or any(not isinstance(author, str) or not author.strip() for author in authors):
        raise UnexpectedArxivResponseError("arXiv result has invalid author metadata.")
    if not isinstance(abstract, str) or not abstract.strip() or not isinstance(pdf_url, str) or not pdf_url:
        raise UnexpectedArxivResponseError("arXiv result has invalid text or PDF metadata.")
    if not categories or any(not isinstance(category, str) for category in categories):
        raise UnexpectedArxivResponseError("arXiv result has invalid category metadata.")
    if not isinstance(entry_id, str):
        raise UnexpectedArxivResponseError("arXiv result has no valid abstract URL.")

    intent = classify_query(entry_id)
    if intent.kind is not QueryKind.PAPER_URL or intent.arxiv_id is None:
        raise UnexpectedArxivResponseError("arXiv result has an invalid abstract URL.")
    if not isinstance(published, datetime):
        raise UnexpectedArxivResponseError("arXiv result has an invalid publication date.")

    return PaperMetadata(
        title=title.strip(),
        authors=authors,
        arxiv_id=intent.arxiv_id,
        abstract=abstract.strip(),
        published=published,
        categories=categories,
        pdf_url=pdf_url,
        abstract_url=f"https://arxiv.org/abs/{intent.arxiv_id}",
    )


class ArxivRetriever:
    """Retrieve papers by arXiv identifier or search topic."""

    def __init__(
        self,
        client: object | None = None,
        search_factory: Callable[..., object] | None = None,
        max_results: int = 10,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        if client is None or search_factory is None:
            try:
                import arxiv
            except ImportError as exc:
                raise ArxivApiError(
                    "The 'arxiv' package is required for live retrieval."
                ) from exc
            if client is None:
                client = arxiv.Client()
            if search_factory is None:
                search_factory = arxiv.Search

        self._client = client
        self._search_factory = search_factory
        self._max_results = max_results

    def retrieve(self, query: str) -> list[PaperMetadata]:
        """Search arXiv and return normalized metadata for each matching paper."""
        intent = classify_query(query)
        if intent.kind is QueryKind.INVALID:
            raise InvalidQueryError(intent.message or "Enter an arXiv ID, URL, or topic.")

        try:
            if intent.kind in (QueryKind.PAPER_ID, QueryKind.PAPER_URL):
                search = self._search_factory(
                    id_list=[intent.arxiv_id], max_results=1
                )
            else:
                search = self._search_factory(
                    query=intent.query, max_results=self._max_results
                )
            results = list(self._client.results(search))
        except Exception as exc:
            raise ArxivApiError("The arXiv API request failed.") from exc

        if not results:
            raise NoResultsError(f"No arXiv papers found for {intent.query!r}.")

        return [paper_metadata_from_result(result) for result in results]
