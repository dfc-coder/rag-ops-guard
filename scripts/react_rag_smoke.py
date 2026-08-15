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
FOLLOWUP = "Y despues del tercero?"
FOLLOWUP_QUERY = "Cuantos reintentos permite Calypso. Y despues del tercero?"


def _contains_retry_rule(text: str) -> bool:
    folded = text.casefold()
    return "maximum of three times" in folded or "up to three times" in folded


def _contains_escalation(text: str) -> bool:
    return "treasury integrations" in text.casefold()


def _print_result(label: str, result) -> None:
    print(
        f"{label}: supported={result.supported} relevance={result.relevance:.4f} "
        f"admitted={[item.chunk.title for item in result.admitted]}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    context = QueryContext()

    print("\n=== retrieval diagnostic: initial Calypso fact ===")
    result = knowledge_search().search(QUERY, context, query_mode="knowledge")
    _print_result("initial", result)
    if not result.supported or not result.admitted:
        raise SystemExit("RAG smoke failed: Calypso evidence was not admitted")

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

    print("\n=== retrieval diagnostic: contextual follow-up ===")
    follow_result = knowledge_search().search(
        FOLLOWUP_QUERY,
        context,
        query_mode="knowledge",
        ranking_query=FOLLOWUP,
    )
    _print_result("follow-up retrieval", follow_result)
    if not follow_result.supported or not follow_result.admitted:
        raise SystemExit("RAG smoke failed: contextual follow-up evidence was not admitted")
    escalation_sources = [item for item in follow_result.admitted if _contains_escalation(item.chunk.text)]
    if not escalation_sources:
        raise SystemExit(
            "RAG smoke failed: contextual retrieval did not admit evidence containing the "
            "Treasury Integrations escalation"
        )
    print(f"authoritative escalation evidence={[item.chunk.title for item in escalation_sources]}")

    print("\n=== policy-driven streaming: exact Chainlit question ===")
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
        f"answer: {terminal.elapsed_ms / 1000:.2f}s · policy={terminal.policy} · "
        f"tools={terminal.tool_calls} · tokens={token_events} · {terminal.text}"
    )
    if terminal.policy != "retrieve":
        raise SystemExit(f"RAG smoke failed: expected retrieve policy, got {terminal.policy}")
    if terminal.tool_calls < 1:
        raise SystemExit("RAG smoke failed: Calypso question did not call search_knowledge")
    if token_events < 1:
        raise SystemExit("RAG smoke failed: grounded response did not stream")
    if not saw_rag_status:
        raise SystemExit("RAG smoke failed: UI stream never exposed RAG activity status")
    if not any(token in terminal.text.casefold() for token in ("3", "tres", "three")):
        raise SystemExit("RAG smoke failed: final answer does not contain retry count")

    state = agent.grounding_state(thread_id)
    print(
        f"grounding state: turn={state.turn_index} grounded={state.grounded} "
        f"topic={state.topic!r} sources={len(state.evidence.sources) if state.evidence else 0}"
    )
    if not state.grounded or state.evidence is None:
        raise SystemExit("RAG smoke failed: successful RAG turn did not commit an evidence window")
    if state.last_grounded_query != QUERY:
        raise SystemExit("RAG smoke failed: initial grounding root query was not preserved")

    print("\n=== new internal fact: contextual follow-up ===")
    followup = agent.invoke(FOLLOWUP, thread_id=thread_id, context=context)
    print(
        f"follow-up: {followup.elapsed_ms / 1000:.2f}s · policy={followup.policy} · "
        f"tools={followup.tool_calls} · {followup.answer}"
    )
    if followup.failed:
        raise SystemExit(f"RAG smoke failed during follow-up: {followup.answer}")
    if followup.policy != "retrieve":
        raise SystemExit(f"RAG smoke failed: follow-up policy was {followup.policy}, expected retrieve")
    if followup.tool_calls < 1:
        raise SystemExit("RAG smoke failed: contextual operational follow-up was not re-grounded")
    if "treasury integrations" not in followup.answer.casefold():
        raise SystemExit("RAG smoke failed: follow-up lost the Treasury Integrations escalation")

    after_followup = agent.grounding_state(thread_id)
    if after_followup.last_grounded_query != QUERY:
        raise SystemExit(
            "RAG smoke failed: contextual follow-up polluted the stable grounding root query"
        )

    print("\n=== evidence window reuse: transformation only ===")
    reuse = agent.invoke("Resumilo en una linea.", thread_id=thread_id, context=context)
    print(
        f"reuse: {reuse.elapsed_ms / 1000:.2f}s · policy={reuse.policy} · "
        f"tools={reuse.tool_calls} · {reuse.answer}"
    )
    if reuse.failed:
        raise SystemExit(f"RAG smoke failed during evidence reuse: {reuse.answer}")
    if reuse.policy != "reuse_evidence":
        raise SystemExit(f"RAG smoke failed: expected evidence reuse policy, got {reuse.policy}")
    if reuse.tool_calls != 0:
        raise SystemExit("RAG smoke failed: evidence transformation unnecessarily called a tool")

    print(
        "\nCONVERSATIONAL GROUNDING V2 READY: initial retrieval + contextual retrieval + "
        "streaming + stable topic + follow-up re-grounding + evidence reuse passed"
    )


if __name__ == "__main__":
    main()
