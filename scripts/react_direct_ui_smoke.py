from __future__ import annotations

from uuid import uuid4

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext

PROMPT = "Escribe una función corta en Python que sume dos números. Solo devuelve el código."


def main() -> None:
    agent = ReactAgent()
    thread_id = f"direct-ui-smoke-{uuid4()}"
    terminal = None
    token_events = 0
    first_status = None

    for event in agent.stream(PROMPT, thread_id=thread_id, context=QueryContext()):
        if event.kind == "status" and first_status is None:
            first_status = event.text
        elif event.kind == "token":
            token_events += 1
        elif event.kind in {"done", "error"}:
            terminal = event

    if first_status is None:
        raise SystemExit("Direct UI smoke failed: no immediate status event")
    if terminal is None or terminal.kind == "error":
        raise SystemExit(f"Direct UI smoke failed: terminal={terminal}")
    if token_events < 1:
        raise SystemExit("Direct UI smoke failed: no streamed tokens")
    if terminal.tool_calls != 0:
        raise SystemExit("Direct UI smoke failed: coding request unexpectedly used RAG")
    if "def " not in terminal.text:
        raise SystemExit("Direct UI smoke failed: response does not contain Python code")

    print(
        "DIRECT UI READY: "
        f"status={first_status!r} · tokens={token_events} · "
        f"tools={terminal.tool_calls} · total={terminal.elapsed_ms / 1000:.2f}s"
    )


if __name__ == "__main__":
    main()
