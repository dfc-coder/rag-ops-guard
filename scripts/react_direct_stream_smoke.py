from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext

PROMPT = "Escribe dos funciones en código Perl para encode y decode de un Caesar cipher."


def main() -> None:
    agent = ReactAgent()
    thread_id = f"react-direct-stream-{uuid4()}"
    started = perf_counter()
    first_token_ms: int | None = None
    terminal = None

    for event in agent.stream(PROMPT, thread_id=thread_id, context=QueryContext()):
        if event.kind == "status":
            print(f"status: {event.text}")
        elif event.kind == "token" and first_token_ms is None:
            first_token_ms = int((perf_counter() - started) * 1000)
            print(f"first token: {first_token_ms / 1000:.2f}s")
        elif event.kind in {"done", "error"}:
            terminal = event

    if terminal is None:
        raise SystemExit("Direct stream smoke failed: no terminal event")
    if terminal.kind == "error":
        raise SystemExit(f"Direct stream smoke failed: {terminal.text}")
    if first_token_ms is None:
        raise SystemExit("Direct stream smoke failed: response did not stream any visible token")
    if terminal.tool_calls != 0:
        raise SystemExit("Direct stream smoke failed: code request unexpectedly called a RAG tool")

    answer = terminal.text.casefold()
    if answer.count("sub ") < 2:
        raise SystemExit("Direct stream smoke failed: expected two Perl subroutines")

    print(terminal.text)
    print(
        "DIRECT STREAM READY: "
        f"first_token={first_token_ms / 1000:.2f}s · total={terminal.elapsed_ms / 1000:.2f}s · "
        f"tools={terminal.tool_calls}"
    )


if __name__ == "__main__":
    main()
