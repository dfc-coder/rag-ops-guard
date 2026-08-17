from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import httpx
import pytest

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

    with pytest.raises(TimeoutError, match="case-one"):
        with runner.case_wall_timeout("case-one", 0.02):
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
        retrieval_hit_at_5=True,
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
            "retrieval_hit_at_5": True,
            "actual_status": "answered_grounded",
            "actual_sources": ["source"],
            "passed": True,
        }
    ]
