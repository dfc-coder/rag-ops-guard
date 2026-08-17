from __future__ import annotations

import argparse
import json
import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

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
    segment_integrity_ok: bool
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
                self.segment_integrity_ok,
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


def _segment_integrity(payload: dict[str, Any]) -> bool:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return False
    for raw in segments:
        if not isinstance(raw, dict):
            return False
        citations = raw.get("citations")
        cited = isinstance(citations, list) and bool(citations)
        if raw.get("grounded") is not cited:
            return False
    status = str(payload.get("status") or "")
    if status == "answered_grounded":
        return bool(segments) and all(bool(segment.get("citations")) for segment in segments)
    if status == "answered_ungrounded":
        return all(not segment.get("citations") for segment in segments)
    if status == "answered_mixed":
        grounded = [bool(segment.get("citations")) for segment in segments]
        return any(grounded) and not all(grounded)
    return not segments


def require_success(response: httpx.Response, *, case_id: str) -> None:
    if response.is_success:
        return
    body = response.text[:2000]
    raise SystemExit(
        f"golden case {case_id} failed: HTTP {response.status_code}: {body or '<empty body>'}"
    )


def _sample(case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case["id"]),
        "question": str(case["question"]),
        "context": case.get("context", {}),
        "expected_status": str(case["expected_status"]),
        "reference_answer": str(case.get("reference_answer") or ""),
        "payload": payload,
    }


def _case_wall_timeout_seconds() -> float:
    value = float(os.environ.get("GOLDEN_CASE_TIMEOUT_SECONDS", "240"))
    if value < 5.0:
        raise SystemExit("GOLDEN_CASE_TIMEOUT_SECONDS must be at least 5 seconds")
    return value


@contextmanager
def case_wall_timeout(case_id: str, seconds: float) -> Iterator[None]:
    """Hard Linux wall-clock bound around the whole case, including retrieval checks."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def on_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"golden case {case_id} exceeded hard wall timeout of {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, on_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def run_case(base_url: str, case: dict[str, Any]) -> tuple[Result, dict[str, Any]]:
    case_id = str(case["id"])
    print(f"GOLDEN {case_id} PHASE agent-query", flush=True)
    try:
        response = httpx.post(
            f"{base_url}/v1/query",
            json={"question": case["question"], "context": case.get("context", {})},
            timeout=float(os.environ.get("GOLDEN_HTTP_TIMEOUT_SECONDS", "180")),
        )
    except httpx.TimeoutException as exc:
        raise SystemExit(
            f"golden case {case_id} timed out during agent-query: {exc}"
        ) from exc
    require_success(response, case_id=case_id)
    payload = response.json()
    citations = [Citation.model_validate(item) for item in payload.get("citations", [])]
    actual_source_ids = (
        set().union(*(citation_identities(item) for item in citations)) if citations else set()
    )
    answer = str(payload.get("answer") or "").lower()
    expected_sources = {str(value) for value in case.get("expected_source_ids", [])}
    forbidden_sources = {str(value) for value in case.get("forbidden_source_ids", [])}
    if expected_sources:
        print(f"GOLDEN {case_id} PHASE retrieval-check", flush=True)
        retrieval_ids = retrieved_identities(str(case["question"]))
    else:
        retrieval_ids = set()
    result = Result(
        id=case_id,
        status_ok=payload.get("status") == case["expected_status"],
        sources_ok=expected_sources.issubset(actual_source_ids),
        forbidden_sources_ok=forbidden_sources.isdisjoint(actual_source_ids),
        required_facts_ok=all(
            str(fact).lower() in answer for fact in case.get("required_facts", [])
        ),
        forbidden_facts_ok=all(
            str(fact).lower() not in answer for fact in case.get("forbidden_facts", [])
        ),
        segment_integrity_ok=_segment_integrity(payload),
        retrieval_hit_at_5=(expected_sources.issubset(retrieval_ids) if expected_sources else None),
        actual_status=str(payload.get("status")),
        actual_sources=sorted(actual_source_ids),
    )
    return result, _sample(case, payload)


def _rate(values: list[bool]) -> float:
    return sum(values) / max(1, len(values))


def _write_results(path: Path, results: list[Result]) -> None:
    serialized = [result.__dict__ | {"passed": result.passed} for result in results]
    path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _select_cases(cases: list[dict[str, Any]], case_id: str | None) -> list[dict[str, Any]]:
    if case_id is None:
        return cases
    selected = [case for case in cases if str(case.get("id")) == case_id]
    if not selected:
        raise SystemExit(f"golden case not found: {case_id}")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        help="Run exactly one golden case for physical diagnostics; aggregate gates are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_cases = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    cases = _select_cases(all_cases, args.case_id)
    thresholds = yaml.safe_load(Path("evaluation/thresholds.yaml").read_text())
    base_url = api_url()
    wall_timeout_seconds = _case_wall_timeout_seconds()

    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "results.partial.json"
    samples_partial_path = output / "golden-samples.partial.json"
    results: list[Result] = []
    samples: list[dict[str, Any]] = []

    print(
        f"GOLDEN START: {len(cases)} case(s); hard-timeout={wall_timeout_seconds:g}s/case",
        flush=True,
    )
    suite_started = perf_counter()
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        started = perf_counter()
        print(f"GOLDEN [{index}/{len(cases)}] START {case_id}", flush=True)
        try:
            with case_wall_timeout(case_id, wall_timeout_seconds):
                result, sample = run_case(base_url, case)
        except TimeoutError as exc:
            elapsed = perf_counter() - started
            print(f"GOLDEN [{index}/{len(cases)}] TIMEOUT {case_id} ({elapsed:.1f}s)", flush=True)
            raise SystemExit(str(exc)) from exc
        except BaseException:
            elapsed = perf_counter() - started
            print(f"GOLDEN [{index}/{len(cases)}] ERROR {case_id} ({elapsed:.1f}s)", flush=True)
            raise
        results.append(result)
        samples.append(sample)
        _write_results(partial_path, results)
        _write_json(samples_partial_path, samples)
        elapsed = perf_counter() - started
        verdict = "PASS" if result.passed else "FAIL"
        print(
            f"GOLDEN [{index}/{len(cases)}] {verdict} {case_id} "
            f"status={result.actual_status} ({elapsed:.1f}s)",
            flush=True,
        )

    if args.case_id is not None:
        result = results[0]
        _write_json(output / "golden-sample.diagnostic.json", samples[0])
        print(json.dumps(result.__dict__ | {"passed": result.passed}, indent=2), flush=True)
        if not result.passed:
            raise SystemExit(f"golden diagnostic case failed: {result.id}")
        return

    by_id = {result.id: result for result in results}
    retrieval_cases = [item for item in results if item.retrieval_hit_at_5 is not None]
    answered_cases = [
        case for case in all_cases if str(case["expected_status"]).startswith("answered_")
    ]
    sourced_cases = [case for case in all_cases if case.get("expected_source_ids")]
    safety_cases = [case for case in all_cases if case["category"] == "safety"]
    injection_cases = [case for case in all_cases if case["category"] == "prompt_injection"]

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
        "segment_integrity": _rate([item.segment_integrity_ok for item in results]),
        "critical_safety_pass_rate": _rate([by_id[case["id"]].passed for case in safety_cases]),
        "prompt_injection_pass_rate": _rate([by_id[case["id"]].passed for case in injection_cases]),
        "elapsed_seconds": round(perf_counter() - suite_started, 3),
    }
    _write_results(output / "results.json", results)
    _write_json(output / "golden-samples.json", samples)
    _write_json(output / "summary.json", summary)
    partial_path.unlink(missing_ok=True)
    samples_partial_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2), flush=True)

    failures = []
    for metric in (
        "status_accuracy",
        "retrieval_hit_at_5",
        "citation_validity",
        "segment_integrity",
        "critical_safety_pass_rate",
        "prompt_injection_pass_rate",
    ):
        if summary[metric] < float(thresholds[metric]):
            failures.append(f"{metric}: {summary[metric]:.3f} < {thresholds[metric]:.3f}")
    if failures:
        raise SystemExit("evaluation thresholds failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
