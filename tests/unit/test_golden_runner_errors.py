from __future__ import annotations

import importlib.util
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
