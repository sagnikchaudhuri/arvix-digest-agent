"""Deterministic extractive briefing with an optional single-pass summary call."""

from collections import Counter
from dataclasses import dataclass
import json
import math
import re

from arxiv_digest.indexing import PaperChunk, TextChunker
from arxiv_digest.parsing.models import ParsedPaper
from arxiv_digest.retrieval.models import PaperMetadata
from arxiv_digest.summarization.models import (
    NOT_FOUND,
    Evidence,
    PaperBriefing,
)
from arxiv_digest.summarization.provider import LLMProvider, create_llm_provider


SUMMARY_OUTPUT_TOKENS = 300
SUMMARY_TIMEOUT_SECONDS = 90.0
SUMMARY_PASSAGE_CHAR_BUDGET = 3500
_WORD_RE = re.compile(r"[a-z][a-z0-9'-]{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOP_WORDS = frozenset(
    "about after again against all also although among an and any are as at be been "
    "before being between both but by can could did do does each for from further had "
    "has have having he her here hers him his how however i if in into is it its itself "
    "may might more most must my neither no nor not of off on once only or other our out "
    "over own same she should so some such than that the their theirs them themselves "
    "then there these they this those through to too under until up us very was we were "
    "what when where which while who whom why will with would you your".split()
)


class MalformedSummarizationOutput(ValueError):
    """Retained for callers that want to classify invalid summary-provider output."""


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    chunk: PaperChunk
    position: int
    score: float = 0.0


class PaperSummarizer:
    """Build a source-grounded briefing without per-chunk LLM calls."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self._provider = provider
        self._chunker = chunker or TextChunker()

    def summarize(
        self,
        paper_metadata: PaperMetadata,
        parsed_paper: ParsedPaper,
    ) -> PaperBriefing:
        if paper_metadata.arxiv_id != parsed_paper.paper.arxiv_id:
            raise ValueError("paper metadata and parsed paper IDs do not match")

        chunks = self._chunker.chunk(parsed_paper)
        if not chunks:
            return self._empty_briefing(paper_metadata)

        sentences = self._rank_sentences(paper_metadata, parsed_paper, chunks)
        selected_chunks = self._representative_chunks(sentences, chunks)
        fields: dict[str, _Sentence | None] = {
            "problem": self._best_for_field(sentences, "problem"),
            "method": self._best_for_field(sentences, "method"),
            "key_results": self._best_for_field(sentences, "key_results"),
            "limitations": self._best_for_field(sentences, "limitations"),
        }

        fallback_summary = self._extractive_summary(sentences, fields)
        summary = self._single_pass_summary(
            paper_metadata,
            parsed_paper,
            selected_chunks,
            fallback_summary,
        )

        evidence: list[Evidence] = []
        summary_evidence_ids = {chunk.chunk_id for chunk in selected_chunks}
        for chunk in chunks:
            if chunk.chunk_id in summary_evidence_ids:
                evidence.append(self._evidence("summary", chunk))
        for field, sentence in fields.items():
            if sentence is not None:
                evidence.append(self._evidence(field, sentence.chunk))

        self._validate_evidence(
            evidence,
            {chunk.chunk_id: chunk for chunk in chunks},
            {page.page_number for page in parsed_paper.pages},
        )
        return PaperBriefing(
            title=paper_metadata.title,
            authors=paper_metadata.authors,
            arxiv_id=paper_metadata.arxiv_id,
            publication_date=paper_metadata.published.date().isoformat(),
            arxiv_link=paper_metadata.abstract_url,
            plain_english_summary=summary,
            problem=fields["problem"].text if fields["problem"] else NOT_FOUND,
            method=fields["method"].text if fields["method"] else NOT_FOUND,
            key_results=fields["key_results"].text if fields["key_results"] else NOT_FOUND,
            limitations=fields["limitations"].text if fields["limitations"] else NOT_FOUND,
            suggested_follow_up_questions=self._suggested_questions(fields),
            evidence=tuple(evidence),
        )

    @staticmethod
    def _tokens(text: str) -> tuple[str, ...]:
        return tuple(token for token in _WORD_RE.findall(text.casefold()) if token not in _STOP_WORDS)

    def _rank_sentences(
        self,
        metadata: PaperMetadata,
        parsed: ParsedPaper,
        chunks: tuple[PaperChunk, ...],
    ) -> tuple[_Sentence, ...]:
        raw: list[_Sentence] = []
        seen: set[str] = set()
        for position, chunk in enumerate(chunks):
            for piece in _SENTENCE_RE.split(chunk.text.replace("\n", " ")):
                sentence = " ".join(piece.split()).strip()
                normalized = sentence.casefold()
                if len(sentence) < 25 or normalized in seen:
                    continue
                seen.add(normalized)
                raw.append(_Sentence(sentence, chunk, position))

        if not raw:
            return ()
        document_frequency = Counter(
            token for sentence in raw for token in set(self._tokens(sentence.text))
        )
        query_terms = set(self._tokens(" ".join((metadata.title, metadata.abstract, parsed.abstract or ""))))
        total = len(raw)
        ranked: list[_Sentence] = []
        for candidate in raw:
            terms = set(self._tokens(candidate.text))
            relevance = sum(
                math.log1p((total + 1) / (document_frequency[token] + 1))
                for token in terms & query_terms
            )
            section = (candidate.chunk.section or "").casefold()
            section_weight = 1.2 if section else 0.0
            early_position = 1.0 / (1.0 + candidate.position / 6)
            ranked.append(
                _Sentence(
                    candidate.text,
                    candidate.chunk,
                    candidate.position,
                    relevance / math.sqrt(max(len(terms), 1)) + section_weight + early_position,
                )
            )
        return tuple(ranked)

    @staticmethod
    def _role_match(sentence: _Sentence, field: str) -> float:
        section = (sentence.chunk.section or "").casefold()
        text = sentence.text.casefold()
        rules = {
            "problem": (
                ("abstract", "introduction", "background", "motivation"),
                ("problem", "challenge", "aim", "goal", "address", "we ask", "we study"),
            ),
            "method": (
                ("method", "approach", "methodology", "model", "architecture", "design"),
                ("we propose", "we introduce", "our method", "our approach", "algorithm", "architecture"),
            ),
            "key_results": (
                ("result", "experiment", "evaluation", "discussion", "finding"),
                ("results show", "we find", "we show", "outperform", "improv", "achiev", "accuracy", "performance"),
            ),
            "limitations": (
                ("limitation", "future work", "conclusion"),
                ("limitation", "limited", "future work", "remains open", "cannot", "not address"),
            ),
        }
        sections, cues = rules[field]
        score = 3.0 if any(term in section for term in sections) else 0.0
        score += 1.5 if any(term in text for term in cues) else 0.0
        if field == "limitations" and not (score and any(term in text for term in cues)):
            return 0.0
        return score

    def _best_for_field(self, sentences: tuple[_Sentence, ...], field: str) -> _Sentence | None:
        eligible = [item for item in sentences if self._role_match(item, field) > 0]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (self._role_match(item, field) + item.score, -item.position, item.text),
        )

    def _representative_chunks(
        self,
        sentences: tuple[_Sentence, ...],
        chunks: tuple[PaperChunk, ...],
    ) -> tuple[PaperChunk, ...]:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        selected_ids: list[str] = []
        for field in ("problem", "method", "key_results", "limitations"):
            candidate = self._best_for_field(sentences, field)
            if candidate and candidate.chunk.chunk_id not in selected_ids:
                selected_ids.append(candidate.chunk.chunk_id)

        ranked_chunks: list[tuple[float, int, PaperChunk]] = []
        for position, chunk in enumerate(chunks):
            matching = [item for item in sentences if item.chunk.chunk_id == chunk.chunk_id]
            section = (chunk.section or "").casefold()
            coverage = sum(
                1 for terms in ("abstract introduction", "method approach model", "result experiment discussion", "conclusion")
                if any(term in section for term in terms.split())
            )
            score = max((item.score for item in matching), default=0.0) + coverage * 1.5
            ranked_chunks.append((score, -position, chunk))

        for _, _, chunk in sorted(ranked_chunks, reverse=True):
            if chunk.chunk_id not in selected_ids:
                selected_ids.append(chunk.chunk_id)

        chosen: list[PaperChunk] = []
        size = 0
        for chunk_id in selected_ids:
            chunk = by_id[chunk_id]
            cost = len(chunk.text) + len(chunk.chunk_id) + 24
            if chosen and size + cost > SUMMARY_PASSAGE_CHAR_BUDGET:
                continue
            if not chosen and cost > SUMMARY_PASSAGE_CHAR_BUDGET:
                chosen.append(chunk)
                break
            chosen.append(chunk)
            size += cost
        return tuple(chosen)

    @staticmethod
    def _extractive_summary(
        sentences: tuple[_Sentence, ...],
        fields: dict[str, _Sentence | None],
    ) -> str:
        selected: list[str] = []
        for field in ("problem", "method", "key_results"):
            item = fields[field]
            if item and item.text not in selected:
                selected.append(item.text)
        if not selected:
            selected = [item.text for item in sorted(sentences, key=lambda item: (-item.score, item.position))[:3]]
        summary = " ".join(selected[:3]).strip()
        return summary[:1200].rstrip() or NOT_FOUND

    def _single_pass_summary(
        self,
        metadata: PaperMetadata,
        parsed: ParsedPaper,
        chunks: tuple[PaperChunk, ...],
        fallback: str,
    ) -> str:
        if not chunks:
            return fallback
        provider = self._provider
        if provider is None:
            try:
                provider = create_llm_provider()
            except Exception:
                return fallback

        passages = "\n\n".join(
            f"CHUNK_ID: {chunk.chunk_id}\nSECTION: {chunk.section or 'Unlabeled'}\n"
            f"PAGES: {', '.join(map(str, chunk.page_numbers))}\nPASSAGE:\n{chunk.text}"
            for chunk in chunks
        )
        prompt = (
            "Write a short plain-English summary using only the paper excerpts below. "
            "Do not add facts, results, methods, or limitations absent from the excerpts. "
            "Return JSON only in the form {\"summary\": \"...\"}. Keep it under 120 words.\n"
            f"TITLE: {metadata.title}\n"
            f"ABSTRACT CONTEXT: {(parsed.abstract or metadata.abstract)[:700]}\n"
            f"EXCERPTS:\n{passages}"
        )
        try:
            limited_generate = getattr(provider, "generate_with_options", None)
            if callable(limited_generate):
                output = limited_generate(
                    prompt,
                    max_output_tokens=SUMMARY_OUTPUT_TOKENS,
                    timeout_seconds=SUMMARY_TIMEOUT_SECONDS,
                )
            else:
                output = provider.generate(prompt)
            payload = json.loads(output)
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if not isinstance(summary, str) or not summary.strip():
                return fallback
            return summary.strip()
        except Exception:
            return fallback

    @staticmethod
    def _evidence(field: str, chunk: PaperChunk) -> Evidence:
        return Evidence(
            field=field,
            quote=chunk.text,
            page_numbers=chunk.page_numbers,
            chunk_id=chunk.chunk_id,
        )

    @staticmethod
    def _validate_evidence(
        evidence: list[Evidence],
        chunks_by_id: dict[str, PaperChunk],
        parsed_page_numbers: set[int],
    ) -> None:
        for item in evidence:
            source = chunks_by_id.get(item.chunk_id or "")
            if (
                source is None
                or item.quote != source.text
                or item.page_numbers != source.page_numbers
                or not set(item.page_numbers) <= parsed_page_numbers
            ):
                raise MalformedSummarizationOutput(
                    "application-generated evidence does not match parsed chunk provenance"
                )

    @staticmethod
    def _suggested_questions(fields: dict[str, _Sentence | None]) -> tuple[str, ...]:
        questions = ["How does the paper evaluate its proposed approach?"]
        if fields["limitations"] is not None:
            questions.append("What limitations do the authors identify?")
        else:
            questions.append("What additional experiments could test the paper's claims?")
        return tuple(questions)

    @staticmethod
    def _empty_briefing(metadata: PaperMetadata) -> PaperBriefing:
        return PaperBriefing(
            title=metadata.title,
            authors=metadata.authors,
            arxiv_id=metadata.arxiv_id,
            publication_date=metadata.published.date().isoformat(),
            arxiv_link=metadata.abstract_url,
            plain_english_summary=NOT_FOUND,
            problem=NOT_FOUND,
            method=NOT_FOUND,
            key_results=NOT_FOUND,
            limitations=NOT_FOUND,
            suggested_follow_up_questions=("Which sections of the paper should be extracted or reviewed?",),
            evidence=(),
        )
