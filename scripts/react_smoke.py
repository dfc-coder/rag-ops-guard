from __future__ import annotations

from uuid import uuid4

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext


def main() -> None:
    agent = ReactAgent()
    thread_id = f"react-smoke-{uuid4()}"

    hello = agent.invoke("Hola", thread_id=thread_id, context=QueryContext())
    print(f"chat: {hello.elapsed_ms / 1000:.2f}s · tools={hello.tool_calls} · {hello.answer}")
    if hello.tool_calls != 0:
        raise SystemExit("ReAct smoke failed: greeting unexpectedly called a tool")

    grounded = agent.invoke(
        "Cuantos reintentos permite Calypso?",
        thread_id=thread_id,
        context=QueryContext(environment="production"),
    )
    print(
        f"rag: {grounded.elapsed_ms / 1000:.2f}s · tools={grounded.tool_calls} · {grounded.answer}"
    )
    if grounded.tool_calls < 1:
        raise SystemExit("ReAct smoke failed: operational question did not call the RAG tool")
    if not any(token in grounded.answer.casefold() for token in ("3", "tres", "three")):
        raise SystemExit("ReAct smoke failed: grounded answer did not contain the retry count")

    followup = agent.invoke(
        "Y despues del tercero?",
        thread_id=thread_id,
        context=QueryContext(environment="production"),
    )
    print(
        f"follow-up: {followup.elapsed_ms / 1000:.2f}s · tools={followup.tool_calls} · "
        f"{followup.answer}"
    )
    if "treasury integrations" not in followup.answer.casefold():
        raise SystemExit("ReAct smoke failed: follow-up lost the Treasury escalation context")

    print("REACT READY: conversation + RAG tool + follow-up memory passed")


if __name__ == "__main__":
    main()
