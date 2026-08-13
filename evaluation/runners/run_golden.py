from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from rag_ops_guard.app import embeddings, vector_store
from rag_ops_guard.domain.models import Citation
from rag_ops_guard.retrieval.query_instruction import embedding_query


@dataclass
class Result:
    id: str
    status_ok: bool
    sources_ok: bool
    forbidden_sources_ok: bool
    required_facts_ok: bool
    forbidden_facts_ok: bool
    retrieval_hit_at_5: bool | None
    actual_status: str
    actual_sources: list[str]

    @property
    def passed(self) -> bool:
        return all(
            (
                self.status_ok,
                self.sources_ok,
                self.forbidden_sources_ok,
                self.required_facts_ok,
                self.forbidden_facts_ok,
            )
        )


def api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def citation_identities(citation: Citation) -> set[str]:
    major = citation.version.split(".", maxsplit=1)[0]
    return {citation.logical_id, f"{citation.logical_id}-v{major}"}


def retrieved_identities(question: str) -> set[str]:
    vector = embeddings().embed_query(embedding_query(question))
    evidence = vector_store().query(vector, top_k=5)
    identities: set[str] = set()
    for item in evidence:
        major = item.chunk.version.split(".", maxsplit=1)[0]
        identities.add(item.chunk.logical_id)
        identities.add(f"{item.chunk.logical_id}-v{major}")
        identities.add(item.chunk.metadata.id)
    return identities


def run_case(base_url: str, case: dict[str, Any]) -> Result:
    response = httpx.post(
        f"{base_url}/v1/query",
        json={"question": case["question"], "context": case.get("context", {})},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    citations = [Citation.model_validate(item) for item in payload.get("citations", [])]
    actual_source_ids = (
        set().union(*(citation_identities(item) for item in citations)) if citations else set()
    )
    answer = str(payload.get("answer") or "").lower()
    expected_sources = {str(value) for value in case.get("expected_source_ids", [])}
    forbidden_sources = {str(value) for value in case.get("forbidden_source_ids", [])}
    retrieval_ids = retrieved_identities(str(case["question"])) if expected_sources else set()
    return Result(
        id=str(case["id"]),
        status_ok=payload.get("status") == case["expected_status"],
        sources_ok=expected_sources.issubset(actual_source_ids),
        forbidden_sources_ok=forbidden_sources.isdisjoint(actual_source_ids),
        required_facts_ok=all(
            str(fact).lower() in answer for fact in case.get("required_facts", [])
        ),
        forbidden_facts_ok=all(
            str(fact).lower() not in answer for fact in case.get("forbidden_facts", [])
        ),
        retrieval_hit_at_5=(expected_sources.issubset(retrieval_ids) if expected_sources else None),
        actual_status=str(payload.get("status")),
        actual_sources=sorted(actual_source_ids),
    )


def _rate(values: list[bool]) -> float:
    return sum(values) / max(1, len(values))


def main() -> None:
    cases = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    thresholds = yaml.safe_load(Path("evaluation/thresholds.yaml").read_text())
    base_url = api_url()
    results = [run_case(base_url, case) for case in cases]
    by_id = {result.id: result for result in results}

    retrieval_cases = [item for item in results if item.retrieval_hit_at_5 is not None]
    answered_cases = [case for case in cases if case["expected_status"] == "answered"]
    sourced_cases = [case for case in cases if case.get("expected_source_ids")]
    safety_cases = [case for case in cases if case["category"] == "safety"]
    injection_cases = [case for case in cases if case["category"] == "prompt_injection"]

    summary = {
        "cases": len(results),
        "passed": sum(result.passed for result in results),
        "case_accuracy": _rate([item.passed for item in results]),
        "status_accuracy": _rate([item.status_ok for item in results]),
        "answer_fact_accuracy": _rate(
            [
                by_id[case["id"]].required_facts_ok and by_id[case["id"]].forbidden_facts_ok
                for case in answered_cases
            ]
        ),
        "source_accuracy": _rate(
            [
                by_id[case["id"]].sources_ok and by_id[case["id"]].forbidden_sources_ok
                for case in sourced_cases
            ]
        ),
        "retrieval_hit_at_5": _rate([bool(item.retrieval_hit_at_5) for item in retrieval_cases]),
        "citation_validity": _rate([item.forbidden_sources_ok for item in results]),
        "critical_safety_pass_rate": _rate([by_id[case["id"]].passed for case in safety_cases]),
        "prompt_injection_pass_rate": _rate(
            [by_id[case["id"]].passed for case in injection_cases]
        ),
    }
    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    serialized = [r.__dict__ | {"passed": r.passed} for r in results]
    (output / "results.json").write_text(json.dumps(serialized, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    failures = []
    for metric in (
        "status_accuracy",
        "retrieval_hit_at_5",
        "citation_validity",
        "critical_safety_pass_rate",
        "prompt_injection_pass_rate",
    ):
        if summary[metric] < float(thresholds[metric]):
            failures.append(f"{metric}: {summary[metric]:.3f} < {thresholds[metric]:.3f}")
    if failures:
        raise SystemExit("evaluation thresholds failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
