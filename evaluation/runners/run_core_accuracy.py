from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_ops_guard.app import embeddings, query_workflow, vector_store
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.retrieval.query_instruction import embedding_query

SELECTED_IDS = {
    "retry-count-1",
    "timeout-manual-2",
    "dlq-2",
    "owner-1",
    "sla-1",
    "missing-sap-timeout",
    "ambiguous-payments-1",
    "deprecated-1",
    "prompt-injection-1",
    "safety-1",
}


def _citation_identities(citation: Any) -> set[str]:
    major = citation.version.split(".", maxsplit=1)[0]
    return {citation.logical_id, f"{citation.logical_id}-v{major}"}


def _retrieved_identities(question: str) -> set[str]:
    vector = embeddings().embed_query(embedding_query(question))
    evidence = vector_store().query(vector, top_k=5)
    identities: set[str] = set()
    for item in evidence:
        major = item.chunk.version.split(".", maxsplit=1)[0]
        identities.update(
            {
                item.chunk.logical_id,
                f"{item.chunk.logical_id}-v{major}",
                item.chunk.metadata.id,
            }
        )
    return identities


def _rate(values: list[bool]) -> float:
    return round(sum(values) / max(1, len(values)), 4)


def main() -> None:
    dataset = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    cases = [case for case in dataset if case["id"] in SELECTED_IDS]
    results: list[dict[str, Any]] = []

    for case in cases:
        context = QueryContext.model_validate(case.get("context", {}))
        response = query_workflow().invoke(
            QueryRequest(question=case["question"], context=context)
        )
        source_ids: set[str] = set()
        for citation in response.citations:
            source_ids.update(_citation_identities(citation))
        expected_sources = set(case.get("expected_source_ids", []))
        forbidden_sources = set(case.get("forbidden_source_ids", []))
        answer = (response.answer or "").lower()
        retrieved_ids = _retrieved_identities(case["question"]) if expected_sources else set()

        result = {
            "id": case["id"],
            "expected_status": case["expected_status"],
            "actual_status": response.status.value,
            "status_ok": response.status.value == case["expected_status"],
            "facts_ok": all(
                str(fact).lower() in answer for fact in case.get("required_facts", [])
            )
            and all(
                str(fact).lower() not in answer for fact in case.get("forbidden_facts", [])
            ),
            "sources_ok": expected_sources.issubset(source_ids)
            and forbidden_sources.isdisjoint(source_ids),
            "retrieval_hit_at_5": (
                expected_sources.issubset(retrieved_ids) if expected_sources else None
            ),
            "actual_sources": sorted(source_ids),
        }
        result["case_ok"] = bool(
            result["status_ok"] and result["facts_ok"] and result["sources_ok"]
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    retrieval = [r["retrieval_hit_at_5"] for r in results if r["retrieval_hit_at_5"] is not None]
    summary = {
        "cases": len(results),
        "case_accuracy": _rate([bool(r["case_ok"]) for r in results]),
        "status_accuracy": _rate([bool(r["status_ok"]) for r in results]),
        "fact_accuracy": _rate([bool(r["facts_ok"]) for r in results]),
        "source_accuracy": _rate([bool(r["sources_ok"]) for r in results]),
        "retrieval_hit_at_5": _rate([bool(value) for value in retrieval]),
    }
    output = Path("artifacts/accuracy")
    output.mkdir(parents=True, exist_ok=True)
    (output / "core-results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "core-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
