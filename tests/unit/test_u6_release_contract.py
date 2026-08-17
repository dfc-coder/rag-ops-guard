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


def test_golden_datasets_use_segmented_response_statuses() -> None:
    """SPEC-6: legacy `answered` must not survive the response-contract pivot."""
    cases: list[dict[str, object]] = []
    for path in (
        ROOT / "evaluation/datasets/golden-v1.json",
        ROOT / "evaluation/datasets/golden-mixed-v1.json",
    ):
        cases.extend(json.loads(path.read_text(encoding="utf-8")))
    statuses = {str(case["expected_status"]) for case in cases}
    assert statuses <= ALLOWED_STATUSES
    assert "answered" not in statuses
    assert "answered_mixed" in statuses


def test_double_relevance_calibration_dataset_is_role_named_and_preserved() -> None:
    """SPEC-6: the U4 three-class fixture remains release evidence under one semantic name."""
    path = ROOT / "evaluation/datasets/retrieval-relevance-calibration.json"
    assert path.exists()
    assert not (ROOT / "evaluation/datasets/retrieval-calibration-v1.json").exists()
    assert not (ROOT / "evaluation/datasets/retrieval-calibration-v2.json").exists()
    cases = json.loads(path.read_text(encoding="utf-8"))
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
