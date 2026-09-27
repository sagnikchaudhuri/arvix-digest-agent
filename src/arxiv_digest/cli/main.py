"""Interactive terminal interface for the arXiv digest agent."""

from collections.abc import Callable
from typing import Any

from arxiv_digest import __version__
from arxiv_digest.agent import ArxivDigestAgent


EXIT_COMMANDS = {"exit", "quit", "q"}
RESET_COMMAND = "reset"


def main() -> int:
    return run_cli()


def run_cli(
    agent: Any | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Run the interactive session; injectable I/O and agent keep it testable."""
    output_fn(f"arxiv-digest-agent {__version__}")
    try:
        active_agent = agent or ArxivDigestAgent(
            progress_callback=lambda event: _show_progress(event, output_fn)
        )
    except Exception as exc:
        output_fn(f"Could not initialize the agent: {exc}")
        return 1

    output_fn("Research digest and paper-grounded Q&A")
    try:
        while True:
            query = _read(input_fn, "Research topic, arXiv ID, or URL (exit to quit): ")
            if query is None:
                output_fn("Goodbye.")
                return 0
            command = query.strip().casefold()
            if command in EXIT_COMMANDS:
                output_fn("Goodbye.")
                return 0
            if command == RESET_COMMAND:
                _reset(active_agent, output_fn)
                continue
            if not query.strip():
                output_fn("Please enter a research topic, arXiv ID, or URL.")
                continue

            try:
                state = active_agent.run(query.strip())
            except Exception as exc:
                output_fn(f"Research pipeline failed: {exc}")
                continue

            if state.get("status") != "briefing_ready" or state.get("briefing") is None:
                _show_pipeline_failure(state, output_fn)
                continue
            _show_warnings(state, output_fn)
            _show_briefing(state["briefing"], output_fn)
            action = _qa_loop(active_agent, input_fn, output_fn)
            if action == "exit":
                output_fn("Goodbye.")
                return 0
            if action == "reset":
                _reset(active_agent, output_fn)
    except KeyboardInterrupt:
        output_fn("\nGoodbye.")
        return 130


def _read(input_fn: Callable[[str], str], prompt: str) -> str | None:
    try:
        return input_fn(prompt)
    except EOFError:
        return None


def _qa_loop(
    agent: Any,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> str:
    while True:
        question = _read(input_fn, "\nQuestion (reset for another paper, exit to quit): ")
        if question is None:
            return "exit"
        command = question.strip().casefold()
        if command in EXIT_COMMANDS:
            return "exit"
        if command == RESET_COMMAND:
            return "reset"
        if not question.strip():
            output_fn("Please enter a question, reset, or exit.")
            continue
        try:
            result = agent.ask(question.strip())
        except Exception as exc:
            output_fn(f"Could not answer that question: {exc}")
            _show_session_issues(agent, output_fn)
            continue
        output_fn(f"\nAnswer: {result.answer}")
        _show_answer_sources(result, output_fn)


def _show_briefing(briefing: Any, output_fn: Callable[[str], None]) -> None:
    output_fn("\n" + "=" * 72)
    output_fn(briefing.title)
    output_fn(f"Authors: {', '.join(briefing.authors)}")
    output_fn(f"arXiv: {briefing.arxiv_id} | Published: {briefing.publication_date}")
    output_fn(briefing.arxiv_link)
    output_fn("=" * 72)
    _section("Plain-English summary", briefing.plain_english_summary, output_fn)
    _section("Problem", briefing.problem, output_fn)
    _section("Method / approach", briefing.method, output_fn)
    _section("Key results", briefing.key_results, output_fn)
    _section("Limitations", briefing.limitations, output_fn)
    output_fn("\nSuggested follow-up questions")
    for question in briefing.suggested_follow_up_questions:
        output_fn(f"  - {question}")


def _section(title: str, text: str, output_fn: Callable[[str], None]) -> None:
    output_fn(f"\n{title}")
    output_fn(text)


def _show_progress(event: str, output_fn: Callable[[str], None]) -> None:
    messages = {
        "summarization_started": "Preparing grounded paper briefing...",
        "summarization_completed": "Paper briefing generated.",
        "summarization_failed": "Paper briefing generation failed.",
    }
    message = messages.get(event)
    if message is not None:
        output_fn(message)


def _show_answer_sources(result: Any, output_fn: Callable[[str], None]) -> None:
    if not result.grounded:
        output_fn("Grounded in retrieved paper evidence: no")
        return
    output_fn("Sources")
    for evidence in result.evidence:
        output_fn(
            f"  {evidence.chunk_id} (page {evidence.page_number}): “{evidence.quote}”"
        )


def _show_pipeline_failure(state: Any, output_fn: Callable[[str], None]) -> None:
    status = state.get("status", "failed")
    output_fn(f"\nThe research pipeline stopped: {status.replace('_', ' ')}.")
    issues = state.get("errors", ())
    for issue in issues:
        output_fn(f"  {issue.stage}: {issue.message}")
    _show_warnings(state, output_fn)


def _show_session_issues(agent: Any, output_fn: Callable[[str], None]) -> None:
    state = getattr(agent, "state", None)
    if state is None:
        return
    for issue in state.get("errors", ()):
        output_fn(f"  {issue.stage}: {issue.message}")


def _show_warnings(state: Any, output_fn: Callable[[str], None]) -> None:
    for warning in state.get("warnings", ()):
        output_fn(f"Warning: {warning}")


def _reset(agent: Any, output_fn: Callable[[str], None]) -> None:
    try:
        agent.reset()
    except Exception as exc:
        output_fn(f"Could not reset the session: {exc}")
        return
    output_fn("Session reset. Start with another paper.")


if __name__ == "__main__":
    raise SystemExit(main())
