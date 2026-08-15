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


def _contains_retry_rule(text: str) -> bool:
    folded = text.casefold()
    return "maximum of three times" in folded or "up to three times" in folded


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

    # The acceptance contract is factual, not tied to one exact document title. Both the active
    # Payment Retry Policy and Payments API v2 are high-authority sources for the same retry rule.
    # Requiring a specific logical_id makes the gate brittle even when the admitted evidence is
    # sufficient and authoritative enough to answer the user correctly.
    authoritative_retry_sources = [
        item
        for item in result.admitted
        if item.chunk.metadata.authority >= 90 and _contains_retry_rule(item.chunk.text)
    ]
    if not authoritative_retry_sources:
        raise SystemExit(
            "RAG smoke failed: admitted evidence does not contain the Calypso retry rule "
            "from an authority >= 90 source"
        )
    print(
        "authoritative retry evidence="
        f"{[item.chunk.title for item in authoritative_retry_sources]}"
    )

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
    print(
        f"follow-up: {followup.elapsed_ms / 1000:.2f}s · tools={followup.tool_calls} · "
        f"{followup.answer}"
    )
    if followup.failed:
        raise SystemExit(f"RAG smoke failed during follow-up: {followup.answer}")
    if "treasury integrations" not in followup.answer.casefold():
        raise SystemExit("RAG smoke failed: follow-up lost the Treasury Integrations escalation")

    print("\nREACT RAG READY: retrieval + streaming + tool use + follow-up memory passed")


if __name__ == "__main__":
    main()
