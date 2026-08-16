from __future__ import annotations

import pytest

from rag_ops_guard.evaluation.gates import enforce_metric_thresholds, grounded_segment_text


def test_single_hallucination_fails_gate_despite_passing_mean() -> None:
    """SPEC-5.1"""
    values = {"c0": 0.0, **{f"c{i}": 0.95 for i in range(1, 19)}}

    with pytest.raises(ValueError, match="c0"):
        enforce_metric_thresholds(
            "faithfulness",
            values,
            mean_floor=0.85,
            per_case_floor=0.60,
        )


def test_metric_must_pass_both_mean_and_per_case_floor() -> None:
    """SPEC-5.1"""
    values = {"a": 0.84, "b": 0.86}

    with pytest.raises(ValueError, match="mean"):
        enforce_metric_thresholds(
            "faithfulness",
            values,
            mean_floor=0.86,
            per_case_floor=0.60,
        )


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
    payload = {"segments": [{"text": "General answer", "grounded": False}]}

    assert grounded_segment_text(payload) == ""
