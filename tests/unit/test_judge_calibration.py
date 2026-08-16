from __future__ import annotations

import pytest

from rag_ops_guard.evaluation.gates import (
    HumanJudgeCase,
    JudgePolicy,
    calibrate_judge_policy,
    require_calibration_policy,
)


def _cases(matches: int) -> list[HumanJudgeCase]:
    cases: list[HumanJudgeCase] = []
    for index in range(10):
        human = index % 2 == 0
        judge = human if index < matches else not human
        cases.append(
            HumanJudgeCase(
                case_id=f"c{index}",
                human_pass=human,
                judge_pass=judge,
                judge_score=0.95 if judge else 0.05,
            )
        )
    return cases


def test_judge_calibration_requires_exactly_ten_human_labels() -> None:
    """SPEC-5.3"""
    with pytest.raises(ValueError, match="exactly 10"):
        calibrate_judge_policy(_cases(9)[:9])


def test_high_agreement_can_gate_at_plan_mean_floor() -> None:
    """SPEC-5.3"""
    policy = calibrate_judge_policy(_cases(9))

    assert policy.agreement == 0.9
    assert policy.gating_enabled is True
    assert policy.mean_floor == 0.85


def test_medium_agreement_uses_measured_cutoff_not_conventional_085() -> None:
    """SPEC-5.3"""
    cases = _cases(8)
    cases[0] = HumanJudgeCase("c0", human_pass=True, judge_pass=True, judge_score=0.72)
    policy = calibrate_judge_policy(cases)

    assert policy.agreement == 0.8
    assert policy.gating_enabled is True
    assert policy.mean_floor is not None
    assert policy.mean_floor < 0.85


def test_low_agreement_makes_ragas_informational() -> None:
    """SPEC-5.3"""
    policy = calibrate_judge_policy(_cases(6))

    assert policy.agreement == 0.6
    assert policy.gating_enabled is False
    assert policy.mean_floor is None


def test_missing_calibration_policy_fails_when_release_requires_it() -> None:
    """SPEC-5.3: missing calibration is a release failure by default."""
    with pytest.raises(ValueError, match="calibration absent"):
        require_calibration_policy(None, required=True)


def test_missing_calibration_policy_is_allowed_only_by_explicit_measurement_mode() -> None:
    """SPEC-5.3: bootstrap measurement must opt out explicitly."""
    assert require_calibration_policy(None, required=False) is None


def test_existing_calibration_policy_is_returned_unchanged() -> None:
    """SPEC-5.3"""
    policy = JudgePolicy(
        agreement=0.9,
        gating_enabled=True,
        mean_floor=0.85,
        calibrated_cutoff=0.8,
    )

    assert require_calibration_policy(policy, required=True) is policy
