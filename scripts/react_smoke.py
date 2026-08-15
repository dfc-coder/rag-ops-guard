from __future__ import annotations

import os
from uuid import uuid4

os.environ.setdefault("EMBEDDING_BASE_URL", "http://127.0.0.1:8083/v3")
os.environ.setdefault("EMBEDDING_MODEL", "OpenVINO/Qwen3-Embedding-0.6B-int8-ov")
os.environ.setdefault("RERANKER_BASE_URL", "http://127.0.0.1:8083/v3")
os.environ.setdefault("RERANKER_MODEL", "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov")
os.environ.setdefault("S3_VECTOR_INDEX", "ops-knowledge-openvino-v1")
os.environ.setdefault("RETRIEVAL_TOP_K", "8")
os.environ.setdefault("RETRIEVAL_CONTEXT_K", "3")

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext


def require_success(label: str, failed: bool, answer: str) -> None:
    if failed:
        raise SystemExit(f"ReAct smoke failed during {label}: {answer}")


def main() -> None:
    agent = ReactAgent()
    thread_id = f"react-smoke-{uuid4()}"

    hello = agent.invoke("Hola", thread_id=thread_id, context=QueryContext())
    print(f"chat: {hello.elapsed_ms / 1000:.2f}s · tools={hello.tool_calls} · {hello.answer}")
    require_success("greeting", hello.failed, hello.answer)
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
    require_success("grounded query", grounded.failed, grounded.answer)
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
    require_success("follow-up", followup.failed, followup.answer)
    if "treasury integrations" not in followup.answer.casefold():
        raise SystemExit("ReAct smoke failed: follow-up lost the Treasury escalation context")

    print("REACT READY: conversation + RAG tool + follow-up memory passed")


if __name__ == "__main__":
    main()
