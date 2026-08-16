from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STATUSES = {
    "answered_grounded",
    "answered_mixed",
    "answered_ungrounded",
    "clarification_required",
    "safety_blocked",
    "error",
}


def test_golden_dataset_uses_segmented_response_statuses() -> None:
    """SPEC-6: legacy `answered` must not survive the response-contract pivot."""
    cases = json.loads((ROOT / "evaluation/datasets/golden-v1.json").read_text())
    statuses = {str(case["expected_status"]) for case in cases}
    assert statuses <= ALLOWED_STATUSES
    assert "answered" not in statuses


def test_double_relevance_calibration_dataset_is_preserved() -> None:
    """SPEC-6: U4 calibration fixture must remain part of the release evidence."""
    path = ROOT / "evaluation/datasets/retrieval-calibration-v2.json"
    assert path.exists()
    cases = json.loads(path.read_text())
    assert {case["class"] for case in cases} == {
        "grounded",
        "in_domain_unanswerable",
        "out_of_domain",
    }


def test_acmepay_is_not_product_control_flow() -> None:
    """SPEC-6: AcmePay/Ops names are evaluation data, not orchestration rules."""
    offenders: list[str] = []
    for path in (ROOT / "src/rag_ops_guard").rglob("*.py"):
        text = path.read_text(encoding="utf-8").casefold()
        if "acmepay" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
