"""Errors raised at the arXiv retrieval boundary."""


class RetrievalError(Exception):
    """Base class for expected retrieval failures."""


class InvalidQueryError(RetrievalError):
    """The user input is empty or is a malformed arXiv ID or URL."""


class ArxivApiError(RetrievalError):
    """The arXiv API could not be reached or returned an API failure."""


class NoResultsError(RetrievalError):
    """A valid query returned no papers."""


class UnexpectedArxivResponseError(RetrievalError):
    """The API result did not contain the expected paper metadata."""
