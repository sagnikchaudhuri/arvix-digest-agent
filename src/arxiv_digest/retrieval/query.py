"""Input classification and arXiv identifier/URL parsing."""

from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import unquote, urlparse


class QueryKind(str, Enum):
    PAPER_ID = "paper_id"
    PAPER_URL = "paper_url"
    TOPIC = "topic"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class QueryClassification:
    kind: QueryKind
    query: str
    arxiv_id: str | None = None
    message: str | None = None


_NEW_STYLE_ID = re.compile(r"\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}(?:v[1-9]\d*)?", re.IGNORECASE)
_OLD_STYLE_ID = re.compile(r"[a-z][a-z0-9.-]*/\d{7}(?:v[1-9]\d*)?", re.IGNORECASE)
_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
_URL_PREFIXES = (
    "http://",
    "https://",
    "arxiv.org/",
    "www.arxiv.org/",
    "export.arxiv.org/",
)


def _valid_arxiv_id(value: str) -> bool:
    return bool(_NEW_STYLE_ID.fullmatch(value) or _OLD_STYLE_ID.fullmatch(value))


def parse_arxiv_url(value: str) -> str | None:
    """Return the identifier from a supported arXiv abstract or PDF URL."""
    candidate = value.strip()
    if candidate.startswith(("arxiv.org/", "www.arxiv.org/", "export.arxiv.org/")):
        candidate = f"https://{candidate}"

    try:
        parsed = urlparse(candidate)
        host = parsed.hostname
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or host not in _ARXIV_HOSTS:
        return None

    parts = [part for part in unquote(parsed.path).split("/") if part]
    if len(parts) not in {2, 3} or parts[0] not in {"abs", "pdf"}:
        return None
    identifier = "/".join(parts[1:])
    if parts[0] == "pdf" and identifier.lower().endswith(".pdf"):
        identifier = identifier[:-4]
    return identifier if _valid_arxiv_id(identifier) else None


def classify_query(value: object) -> QueryClassification:
    """Classify a search string without making a network request."""
    if not isinstance(value, str) or not value.strip():
        return QueryClassification(QueryKind.INVALID, "", message="Input is empty.")

    query = value.strip()
    if query.lower().startswith(_URL_PREFIXES):
        identifier = parse_arxiv_url(query)
        if identifier is None:
            return QueryClassification(
                QueryKind.INVALID, query, message="Malformed or unsupported arXiv URL."
            )
        return QueryClassification(QueryKind.PAPER_URL, query, identifier)

    if _valid_arxiv_id(query):
        return QueryClassification(QueryKind.PAPER_ID, query, query)

    if re.match(r"^\d{4}\.|^[a-z][a-z0-9.-]*/", query, re.IGNORECASE):
        return QueryClassification(
            QueryKind.INVALID, query, message="Malformed arXiv identifier."
        )
    if "://" in query:
        return QueryClassification(
            QueryKind.INVALID, query, message="Only arXiv URLs are supported."
        )

    return QueryClassification(QueryKind.TOPIC, query)
