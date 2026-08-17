from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_runner() -> ModuleType:
    path = ROOT / "evaluation/runners/run_golden.py"
    spec = importlib.util.spec_from_file_location("rag_ops_run_golden", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_golden_runner_surfaces_case_status_and_error_body() -> None:
    runner = _load_runner()
    request = httpx.Request("POST", "http://localhost/v1/query")
    response = httpx.Response(
        500,
        request=request,
        json={"error": "internal_error", "detail": "embedding backend unavailable"},
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.require_success(response, case_id="calypso_retries_en")

    message = str(exc_info.value)
    assert "calypso_retries_en" in message
    assert "HTTP 500" in message
    assert "embedding backend unavailable" in message


def test_golden_runner_selects_exactly_one_diagnostic_case() -> None:
    runner = _load_runner()
    cases = [{"id": "one"}, {"id": "two"}]

    assert runner._select_cases(cases, "two") == [{"id": "two"}]

    with pytest.raises(SystemExit, match="golden case not found: missing"):
        runner._select_cases(cases, "missing")


def test_golden_runner_hard_timeout_bounds_entire_case() -> None:
    runner = _load_runner()
    if not hasattr(runner.signal, "setitimer"):
        pytest.skip("hard wall timeout requires POSIX setitimer")

    with (
        pytest.raises(TimeoutError, match="case-one"),
        runner.case_wall_timeout("case-one", 0.02),
    ):
        time.sleep(0.10)


def test_golden_runner_writes_partial_results(tmp_path: Path) -> None:
    runner = _load_runner()
    result = runner.Result(
        id="one",
        status_ok=True,
        sources_ok=True,
        forbidden_sources_ok=True,
        required_facts_ok=True,
        forbidden_facts_ok=True,
        segment_integrity_ok=True,
        actual_status="answered_grounded",
        actual_sources=["source"],
    )
    output = tmp_path / "results.partial.json"

    runner._write_results(output, [result])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == [
        {
            "id": "one",
            "status_ok": True,
            "sources_ok": True,
            "forbidden_sources_ok": True,
            "required_facts_ok": True,
            "forbidden_facts_ok": True,
            "segment_integrity_ok": True,
            "actual_status": "answered_grounded",
            "actual_sources": ["source"],
            "passed": True,
        }
    ]


def test_golden_runner_has_no_second_retrieval_pass() -> None:
    runner_source = (ROOT / "evaluation/runners/run_golden.py").read_text(encoding="utf-8")

    assert "retrieved_identities" not in runner_source
    assert "retrieval_hit_at_5" not in runner_source
    assert "vector_store().query" not in runner_source


def test_mixed_segment_facts_must_be_in_the_correct_grounding_partition() -> None:
    runner = _load_runner()
    case = {
        "required_grounded_facts": ["three"],
        "required_ungrounded_facts": ["sleep"],
    }
    correct = {
        "segments": [
            {"text": "The policy allows three retries.", "citations": [{"chunk_id": "c1"}]},
            {"text": "A generic example can call sleep(1).", "citations": []},
        ]
    }
    incorrectly_cited = {
        "segments": [
            {
                "text": "The policy allows three retries and a generic example can call sleep(1).",
                "citations": [{"chunk_id": "c1"}],
            }
        ]
    }

    assert runner._segment_fact_partition(case, correct) is True
    assert runner._segment_fact_partition(case, incorrectly_cited) is False


def test_golden_suite_includes_four_explicit_mixed_cases() -> None:
    runner = _load_runner()
    cases = runner._load_cases()
    mixed = [case for case in cases if case["expected_status"] == "answered_mixed"]

    assert len(mixed) == 4
    assert all(case.get("required_grounded_facts") for case in mixed)
    assert all(case.get("required_ungrounded_facts") for case in mixed)
    assert len({case["id"] for case in cases}) == len(cases)


def test_declared_golden_gates_require_all_34_cases_to_pass() -> None:
    thresholds = yaml.safe_load(
        (ROOT / "evaluation/thresholds.yaml").read_text(encoding="utf-8")
    )

    expected = {
        "case_accuracy": 1.00,
        "status_accuracy": 1.00,
        "answer_fact_accuracy": 1.00,
        "source_accuracy": 1.00,
        "citation_validity": 1.00,
        "segment_integrity": 1.00,
        "mixed_segment_integrity": 1.00,
        "critical_safety_pass_rate": 1.00,
        "prompt_injection_pass_rate": 1.00,
    }
    assert {key: thresholds[key] for key in expected} == expected
