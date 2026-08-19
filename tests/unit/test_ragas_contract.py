from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ops_guard.evaluation.gates import enforce_metric_thresholds, grounded_segment_text
from rag_ops_guard.evaluation.judge import judge_connection_from_env, judge_identity_from_env


def test_single_hallucination_fails_gate_despite_passing_mean() -> None:
    """SPEC-5.1"""
    values = {"c0": 0.0, **{f"c{i}": 0.95 for i in range(1, 19)}}
    with pytest.raises(ValueError, match="c0"):
        enforce_metric_thresholds("faithfulness", values, mean_floor=0.85, per_case_floor=0.60)


def test_metric_must_pass_both_mean_and_per_case_floor() -> None:
    """SPEC-5.1"""
    values = {"a": 0.84, "b": 0.86}
    with pytest.raises(ValueError, match="mean"):
        enforce_metric_thresholds("faithfulness", values, mean_floor=0.86, per_case_floor=0.60)


def test_faithfulness_input_contains_only_grounded_segments() -> None:
    """SPEC-5.2"""
    payload = {
        "segments": [
            {"text": "Document-backed retry count is three.", "grounded": True},
            {"text": "Exponential backoff is generally useful.", "grounded": False},
            {"text": "This runbook is active.", "citations": [{"chunk_id": "c1"}]},
        ]
    }
    text = grounded_segment_text(payload)
    assert "retry count" in text
    assert "runbook is active" in text
    assert "Exponential backoff" not in text


def test_faithfulness_skips_response_without_grounded_segments() -> None:
    """SPEC-5.2"""
    assert grounded_segment_text({"segments": [{"text": "General answer", "grounded": False}]}) == ""


def test_external_judge_may_differ_from_runtime_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "golden.json"
    dataset.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "local-generation-model")
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "external-judge-model")
    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "secret")
    connection = judge_connection_from_env(dataset)
    assert connection.identity.model == "external-judge-model"
    assert connection.identity.provider == "openai"
    assert connection.base_url == "https://judge.example/v1"
    assert connection.api_key == "secret"


def test_external_judge_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "golden.json"
    dataset.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "judge")
    monkeypatch.delenv("RAGAS_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="ragas_judge_base_url"):
        judge_connection_from_env(dataset)
    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge.example/v1")
    with pytest.raises(ValidationError, match="ragas_judge_api_key"):
        judge_connection_from_env(dataset)


def test_judge_identity_changes_when_dataset_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = tmp_path / "golden.json"
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "judge")
    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "secret")
    dataset.write_text("[]", encoding="utf-8")
    first = judge_identity_from_env(dataset)
    dataset.write_text('[{"id":"new"}]', encoding="utf-8")
    second = judge_identity_from_env(dataset)
    assert first.dataset_sha256 != second.dataset_sha256
