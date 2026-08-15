from __future__ import annotations

import logging
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
from rag_ops_guard.app import knowledge_search
from rag_ops_guard.domain.models import QueryContext

QUERY = "Cuantos reintentos permite Calypso?"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    context = QueryContext()

    print("\n=== retrieval diagnostic: default / any environment ===")
    result = knowledge_search().search(QUERY, context, query_mode="knowledge")
    print(
        "supported="
        f"{result.supported} relevance={result.relevance:.4f} "
        f"admitted={[item.chunk.title for item in result.admitted]}"
    )
    if not result.supported or not result.admitted:
        raise SystemExit("RAG smoke failed: Calypso evidence was not admitted")
    evidence_text = "\n".join(item.chunk.text for item in result.admitted).casefold()
    if "three automated retries" not in evidence_text:
        raise SystemExit("RAG smoke failed: admitted evidence does not contain the retry rule")

    print("\n=== ReAct streaming: exact Chainlit question ===")
    agent = ReactAgent()
    thread_id = f"react-rag-smoke-{uuid4()}"
    terminal = None
    token_events = 0
    saw_rag_status = False
    for event in agent.stream(QUERY, thread_id=thread_id, context=context):
        if event.kind == "status":
            print(f"status: {event.text}")
            saw_rag_status = saw_rag_status or "base de conocimiento" in event.text.casefold()
        elif event.kind == "token":
            token_events += 1
        elif event.kind in {"done", "error"}:
            terminal = event

    if terminal is None or terminal.kind == "error":
        raise SystemExit(f"RAG smoke failed: terminal={terminal}")
    print(
        f"answer: {terminal.elapsed_ms / 1000:.2f}s · tools={terminal.tool_calls} · "
        f"tokens={token_events} · {terminal.text}"
    )
    if terminal.tool_calls < 1:
        raise SystemExit("RAG smoke failed: Calypso question did not call search_knowledge")
    if token_events < 1:
        raise SystemExit("RAG smoke failed: grounded response did not stream")
    if not saw_rag_status:
        raise SystemExit("RAG smoke failed: UI stream never exposed RAG activity status")
    if not any(token in terminal.text.casefold() for token in ("3", "tres", "three")):
        raise SystemExit("RAG smoke failed: final answer does not contain retry count")

    print("\n=== conversation memory: follow-up ===")
    followup = agent.invoke(
        "Y despues del tercero?",
        thread_id=thread_id,
        context=context,
    )
    print(f"follow-up: {followup.elapsed_ms / 1000:.2f}s · tools={followup.tool_calls} · {followup.answer}")
    if followup.failed:
        raise SystemExit(f"RAG smoke failed during follow-up: {followup.answer}")
    if "treasury integrations" not in followup.answer.casefold():
        raise SystemExit("RAG smoke failed: follow-up lost the Treasury Integrations escalation")

    print("\nREACT RAG READY: retrieval + streaming + tool use + follow-up memory passed")


if __name__ == "__main__":
    main()
