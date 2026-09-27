from io import StringIO
import unittest
from unittest.mock import patch

from arxiv_digest.agent.state import AgentIssue
from arxiv_digest.cli.main import main, run_cli
from arxiv_digest.indexing.models import PaperChunk
from arxiv_digest.qa import QAEvidence, QAResult
from arxiv_digest.summarization.models import NOT_FOUND, PaperBriefing


def make_briefing() -> PaperBriefing:
    return PaperBriefing(
        title="A Demonstration Paper",
        authors=("A. Author", "B. Researcher"),
        arxiv_id="2501.12345v1",
        publication_date="2025-01-02",
        arxiv_link="https://arxiv.org/abs/2501.12345v1",
        plain_english_summary=NOT_FOUND,
        problem=NOT_FOUND,
        method=NOT_FOUND,
        key_results=NOT_FOUND,
        limitations=NOT_FOUND,
        suggested_follow_up_questions=("What data was used?",),
        evidence=(),
    )


def make_qa_result() -> QAResult:
    chunk = PaperChunk(
        chunk_id="2501.12345v1:0001",
        paper_id="2501.12345v1",
        text="The evaluation used Dataset A.",
        page_numbers=(3,),
        section="Evaluation",
        metadata={},
    )
    return QAResult(
        question="What data was used?",
        answer="The evaluation used Dataset A.",
        source_chunks=(chunk,),
        page_numbers=(3,),
        retrieval_distances=(0.12,),
        evidence=(QAEvidence(chunk.chunk_id, 3, "The evaluation used Dataset A."),),
        grounded=True,
    )


class FakeAgent:
    def __init__(self, states: list[dict[str, object]] | None = None) -> None:
        self.states = states or [
            {"status": "briefing_ready", "briefing": make_briefing(), "errors": [], "warnings": []}
        ]
        self.run_queries: list[str] = []
        self.questions: list[str] = []
        self.reset_count = 0
        self.state: dict[str, object] = {"errors": []}
        self.qa_error: Exception | None = None

    def run(self, query: str) -> dict[str, object]:
        self.run_queries.append(query)
        self.state = self.states[min(len(self.run_queries) - 1, len(self.states) - 1)]
        return self.state

    def ask(self, question: str) -> QAResult:
        self.questions.append(question)
        if self.qa_error:
            raise self.qa_error
        return make_qa_result()

    def reset(self) -> None:
        self.reset_count += 1
        self.state = {"errors": []}


class InputSequence:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)

    def __call__(self, _prompt: str) -> str:
        return next(self.values)


class TestCli(unittest.TestCase):
    def test_application_entry_point_calls_interactive_cli(self) -> None:
        with patch("arxiv_digest.cli.main.run_cli", return_value=0) as run:
            self.assertEqual(main(), 0)
        run.assert_called_once_with()

    def test_briefing_grounded_answer_sources_and_exit(self) -> None:
        output = StringIO()
        agent = FakeAgent()
        code = run_cli(
            agent,
            input_fn=InputSequence(["KV cache compression", "What dataset did they use?", "exit"]),
            output_fn=lambda line: print(line, file=output),
        )

        rendered = output.getvalue()
        self.assertEqual(code, 0)
        self.assertEqual(agent.run_queries, ["KV cache compression"])
        self.assertEqual(agent.questions, ["What dataset did they use?"])
        self.assertIn("A Demonstration Paper", rendered)
        self.assertIn("Plain-English summary", rendered)
        self.assertIn("Answer: The evaluation used Dataset A.", rendered)
        self.assertIn("2501.12345v1:0001 (page 3)", rendered)
        self.assertIn("Goodbye.", rendered)

    def test_default_agent_reports_briefing_progress(self) -> None:
        output = StringIO()

        class ProgressAgent(FakeAgent):
            def __init__(self, progress_callback):
                super().__init__()
                self.progress_callback = progress_callback

            def run(self, query: str) -> dict[str, object]:
                self.progress_callback("summarization_started")
                state = super().run(query)
                self.progress_callback("summarization_completed")
                return state

        with patch(
            "arxiv_digest.cli.main.ArxivDigestAgent",
            side_effect=lambda **kwargs: ProgressAgent(kwargs["progress_callback"]),
        ):
            code = run_cli(
                input_fn=InputSequence(["KV cache compression", "exit"]),
                output_fn=lambda line: print(line, file=output),
            )

        rendered = output.getvalue()
        self.assertEqual(code, 0)
        start = rendered.index("Preparing grounded paper briefing")
        complete = rendered.index("Paper briefing generated.")
        title = rendered.index("A Demonstration Paper")
        self.assertLess(start, complete)
        self.assertLess(complete, title)

    def test_reset_starts_a_new_research_session(self) -> None:
        agent = FakeAgent()
        run_cli(
            agent,
            input_fn=InputSequence(["first topic", "reset", "second topic", "exit"]),
            output_fn=lambda _line: None,
        )
        self.assertEqual(agent.run_queries, ["first topic", "second topic"])
        self.assertEqual(agent.reset_count, 1)

    def test_structured_pipeline_error_is_displayed_and_loop_continues(self) -> None:
        failed = {
            "status": "no_candidates",
            "briefing": None,
            "errors": [AgentIssue("retrieve_papers", "no_candidates", "No papers found.")],
            "warnings": [],
        }
        agent = FakeAgent([failed])
        output = StringIO()
        code = run_cli(
            agent,
            input_fn=InputSequence(["a topic", "quit"]),
            output_fn=lambda line: print(line, file=output),
        )
        self.assertEqual(code, 0)
        self.assertIn("research pipeline stopped: no candidates", output.getvalue())
        self.assertIn("retrieve_papers: No papers found.", output.getvalue())

    def test_question_failure_is_friendly_and_does_not_crash_loop(self) -> None:
        agent = FakeAgent()
        agent.qa_error = RuntimeError("QA graph did not produce a result")
        output = StringIO()
        code = run_cli(
            agent,
            input_fn=InputSequence(["topic", "question", "exit"]),
            output_fn=lambda line: print(line, file=output),
        )
        self.assertEqual(code, 0)
        self.assertIn("Could not answer that question:", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())

    def test_exit_before_research(self) -> None:
        output = StringIO()
        agent = FakeAgent()
        code = run_cli(
            agent,
            input_fn=InputSequence(["q"]),
            output_fn=lambda line: print(line, file=output),
        )
        self.assertEqual(code, 0)
        self.assertEqual(agent.run_queries, [])
        self.assertIn("arxiv-digest-agent", output.getvalue())


if __name__ == "__main__":
    unittest.main()
