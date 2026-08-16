from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

HUMAN_CALIBRATION_CASES = 10
STRONG_JUDGE_AGREEMENT = 0.90
MINIMUM_GATING_AGREEMENT = 0.70
STRONG_AGREEMENT_MEAN_FLOOR = 0.85


@dataclass(frozen=True)
class HumanJudgeCase:
    case_id: str
    human_pass: bool
    judge_pass: bool
    judge_score: float


@dataclass(frozen=True)
class JudgePolicy:
    agreement: float
    gating_enabled: bool
    mean_floor: float | None
    calibrated_cutoff: float | None


def grounded_segment_text(payload: dict[str, Any]) -> str:
    """SPEC-5.2: faithfulness sees only claims that the response marks as grounded."""
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return ""

    grounded: list[str] = []
    for segment in raw_segments:
        if not isinstance(segment, dict):
            continue
        citations = segment.get("citations")
        is_grounded = segment.get("grounded") is True or (
            isinstance(citations, list) and bool(citations)
        )
        text = str(segment.get("text") or "").strip()
        if is_grounded and text:
            grounded.append(text)
    return "\n\n".join(grounded)


def enforce_metric_thresholds(
    metric: str,
    values: dict[str, float],
    *,
    mean_floor: float,
    per_case_floor: float | None = None,
) -> None:
    """SPEC-5.1: a good aggregate cannot hide one catastrophic case."""
    if not values:
        raise ValueError(f"{metric}: no evaluation cases")

    failures: list[str] = []
    mean = fmean(values.values())
    if mean < mean_floor:
        failures.append(f"{metric} mean {mean:.3f} < {mean_floor:.3f}")

    if per_case_floor is not None:
        failures.extend(
            f"{metric} {case_id}: {score:.3f} < {per_case_floor:.3f}"
            for case_id, score in values.items()
            if score < per_case_floor
        )

    if failures:
        raise ValueError("; ".join(failures))


def require_calibration_policy(
    policy: JudgePolicy | None,
    *,
    required: bool,
) -> JudgePolicy | None:
    """SPEC-5.3: release is fail-closed when the human calibration policy is absent."""
    if policy is None and required:
        raise ValueError(
            "RAGAS calibration absent; release gating requires a calibrated judge policy. "
            "Run scripts/calibrate_ragas_judge.py after producing ragas-results.json."
        )
    return policy


def calibrate_judge_policy(cases: list[HumanJudgeCase]) -> JudgePolicy:
    """Derive whether RAGAS may gate from the fixed human-labelled calibration set."""
    if len(cases) != HUMAN_CALIBRATION_CASES or len({case.case_id for case in cases}) != HUMAN_CALIBRATION_CASES:
        raise ValueError(
            f"judge calibration requires exactly {HUMAN_CALIBRATION_CASES} unique human-labelled cases"
        )

    agreement = (
        sum(case.human_pass == case.judge_pass for case in cases)
        / HUMAN_CALIBRATION_CASES
    )
    calibrated_cutoff = _best_score_cutoff(cases)

    if agreement >= STRONG_JUDGE_AGREEMENT:
        return JudgePolicy(
            agreement=agreement,
            gating_enabled=True,
            mean_floor=STRONG_AGREEMENT_MEAN_FLOOR,
            calibrated_cutoff=calibrated_cutoff,
        )
    if agreement >= MINIMUM_GATING_AGREEMENT:
        return JudgePolicy(
            agreement=agreement,
            gating_enabled=True,
            mean_floor=min(calibrated_cutoff, STRONG_AGREEMENT_MEAN_FLOOR - 0.000001),
            calibrated_cutoff=calibrated_cutoff,
        )
    return JudgePolicy(
        agreement=agreement,
        gating_enabled=False,
        mean_floor=None,
        calibrated_cutoff=calibrated_cutoff,
    )


def _best_score_cutoff(cases: list[HumanJudgeCase]) -> float:
    scores = sorted({0.0, 1.0, *(case.judge_score for case in cases)})
    candidates = sorted(
        {
            *scores,
            *(
                (left + right) / 2
                for left, right in zip(scores, scores[1:], strict=False)
            ),
        }
    )
    ranked: list[tuple[int, float]] = []
    for cutoff in candidates:
        correct = sum((case.judge_score >= cutoff) == case.human_pass for case in cases)
        ranked.append((-correct, cutoff))
    _negative_correct, cutoff = min(ranked)
    return cutoff
