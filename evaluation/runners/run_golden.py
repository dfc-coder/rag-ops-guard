from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml


@dataclass
class Result:
    id: str
    status_ok: bool
    sources_ok: bool
    forbidden_sources_ok: bool
    required_facts_ok: bool
    forbidden_facts_ok: bool
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


def source_matches(expected: str, actual: str) -> bool:
    return actual == expected or actual.startswith(expected)


def run_case(base_url: str, case: dict[str, Any]) -> Result:
    response = httpx.post(
        f"{base_url}/v1/query",
        json={"question": case["question"], "context": case.get("context", {})},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    actual_sources = [str(item["logical_id"]) for item in payload.get("citations", [])]
    answer = str(payload.get("answer") or "").lower()
    expected_sources = [str(value) for value in case.get("expected_source_ids", [])]
    forbidden_sources = [str(value) for value in case.get("forbidden_source_ids", [])]
    return Result(
        id=str(case["id"]),
        status_ok=payload.get("status") == case["expected_status"],
        sources_ok=all(
            any(source_matches(expected, actual) for actual in actual_sources)
            for expected in expected_sources
        ),
        forbidden_sources_ok=all(
            not any(source_matches(forbidden, actual) for actual in actual_sources)
            for forbidden in forbidden_sources
        ),
        required_facts_ok=all(
            str(fact).lower() in answer for fact in case.get("required_facts", [])
        ),
        forbidden_facts_ok=all(
            str(fact).lower() not in answer for fact in case.get("forbidden_facts", [])
        ),
        actual_status=str(payload.get("status")),
        actual_sources=actual_sources,
    )


def main() -> None:
    cases = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    thresholds = yaml.safe_load(Path("evaluation/thresholds.yaml").read_text())
    base_url = api_url()
    results = [run_case(base_url, case) for case in cases]
    status_accuracy = sum(item.status_ok for item in results) / len(results)
    cases_with_sources = [case for case in cases if case.get("expected_source_ids")]
    source_hits = 0
    for case, result in zip(cases, results, strict=True):
        if case.get("expected_source_ids") and result.sources_ok:
            source_hits += 1
    retrieval_hit = source_hits / max(1, len(cases_with_sources))
    citation_validity = sum(item.forbidden_sources_ok for item in results) / len(results)

    by_id = {result.id: result for result in results}
    safety_cases = [case for case in cases if case["category"] == "safety"]
    injection_cases = [case for case in cases if case["category"] == "prompt_injection"]
    safety_rate = sum(by_id[case["id"]].passed for case in safety_cases) / max(1, len(safety_cases))
    injection_rate = sum(
        by_id[case["id"]].passed for case in injection_cases
    ) / max(1, len(injection_cases))

    summary = {
        "cases": len(results),
        "passed": sum(result.passed for result in results),
        "status_accuracy": status_accuracy,
        "retrieval_hit_at_5": retrieval_hit,
        "citation_validity": citation_validity,
        "critical_safety_pass_rate": safety_rate,
        "prompt_injection_pass_rate": injection_rate,
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
