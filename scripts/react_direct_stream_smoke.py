from __future__ import annotations

import re
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

    # Always print the model result before validating the content contract. A model-quality
    # failure must be diagnosable without changing the script or digging through server logs.
    print("\n--- model output ---")
    print(terminal.text)
    print("--- end model output ---\n")

    if terminal.kind == "error":
        reason = f" finish_reason={terminal.finish_reason}" if terminal.finish_reason else ""
        raise SystemExit(
            "Direct stream smoke failed: recoverable/incomplete model turn "
            f"after {terminal.elapsed_ms / 1000:.2f}s.{reason}"
        )
    if first_token_ms is None:
        raise SystemExit("Direct stream smoke failed: response did not stream any visible token")
    if terminal.tool_calls != 0:
        raise SystemExit("Direct stream smoke failed: code request unexpectedly called a RAG tool")
    if terminal.finish_reason in {"length", "max_tokens"}:
        raise SystemExit(
            "Direct stream smoke failed: model hit its output budget before finishing "
            f"(finish_reason={terminal.finish_reason})"
        )

    perl_subroutines = re.findall(r"(?im)^\s*sub\s+[A-Za-z_]\w*\s*", terminal.text)
    if len(perl_subroutines) < 2:
        raise SystemExit(
            "Direct stream smoke failed: streaming worked but direct-code quality contract failed "
            f"(expected >=2 Perl subroutines, found {len(perl_subroutines)})"
        )

    # When the model uses a Markdown code fence, require it to be closed. This catches the exact
    # regression where two `sub` declarations existed but the second function was cut mid-body.
    fence_count = terminal.text.count("```")
    if fence_count and fence_count % 2 != 0:
        raise SystemExit(
            "Direct stream smoke failed: generated code block is visibly truncated "
            f"(unclosed Markdown fence, count={fence_count})"
        )

    print(
        "DIRECT STREAM READY: "
        f"first_token={first_token_ms / 1000:.2f}s · total={terminal.elapsed_ms / 1000:.2f}s · "
        f"tools={terminal.tool_calls} · finish_reason={terminal.finish_reason or 'unknown'} · "
        f"perl_subroutines={len(perl_subroutines)}"
    )


if __name__ == "__main__":
    main()
