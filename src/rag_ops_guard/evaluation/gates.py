from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import fmean
from typing import Any

_QWEN3_4B_RE = re.compile(r"qwen3(?:[-_. ]?)4b", re.IGNORECASE)


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


def require_qwen3_4b_judge(model: str) -> None:
    """SPEC-5.3: RAGAS uses the same Qwen3-4B family chosen for runtime."""
    if not _QWEN3_4B_RE.search(model):
        raise ValueError(
            f"SPEC-5.3 requires Qwen3-4B as the runtime/judge model; got {model!r}"
        )


def calibrate_judge_policy(cases: list[HumanJudgeCase]) -> JudgePolicy:
    """Derive whether RAGAS may gate from exactly ten human-labelled comparisons."""
    if len(cases) != 10 or len({case.case_id for case in cases}) != 10:
        raise ValueError("judge calibration requires exactly 10 unique human-labelled cases")

    agreement = sum(case.human_pass == case.judge_pass for case in cases) / 10
    calibrated_cutoff = _best_score_cutoff(cases)

    if agreement >= 0.9:
        return JudgePolicy(
            agreement=agreement,
            gating_enabled=True,
            mean_floor=0.85,
            calibrated_cutoff=calibrated_cutoff,
        )
    if agreement >= 0.7:
        return JudgePolicy(
            agreement=agreement,
            gating_enabled=True,
            mean_floor=min(calibrated_cutoff, 0.849999),
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
