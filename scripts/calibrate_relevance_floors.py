from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/datasets/retrieval-calibration-v2.json"
VALID_CLASSES = {"grounded", "in_domain_unanswerable", "out_of_domain"}


@dataclass(frozen=True)
class Observation:
    case_id: str
    case_class: str
    domain_score: float
    grounded_score: float

    @property
    def should_be_domain_related(self) -> bool:
        return self.case_class in {"grounded", "in_domain_unanswerable"}

    @property
    def should_be_grounded(self) -> bool:
        return self.case_class == "grounded"


@dataclass(frozen=True)
class CalibrationFloors:
    domain_floor: float
    grounded_floor: float
    domain_false_positives: int
    grounded_false_positives: int
    domain_recall: float
    grounded_recall: float


def calibrate_floors(observations: list[Observation]) -> CalibrationFloors:
    if not observations:
        raise ValueError("calibration requires observations")
    unknown = sorted({item.case_class for item in observations} - VALID_CLASSES)
    if unknown:
        raise ValueError(f"unknown calibration classes: {unknown}")

    _require_domain_class_separation(observations)
    domain_floor, domain_tp, domain_fp, _domain_tn, domain_fn = _best_threshold(
        [(item.domain_score, item.should_be_domain_related) for item in observations]
    )
    grounded_floor, grounded_tp, grounded_fp, _grounded_tn, grounded_fn = _best_threshold(
        [(item.grounded_score, item.should_be_grounded) for item in observations]
    )
    domain_recall = _recall(domain_tp, domain_fn)
    grounded_recall = _recall(grounded_tp, grounded_fn)
    return CalibrationFloors(
        domain_floor=domain_floor,
        grounded_floor=grounded_floor,
        domain_false_positives=domain_fp,
        grounded_false_positives=grounded_fp,
        domain_recall=domain_recall,
        grounded_recall=grounded_recall,
    )


def expected_grounded_score(result: Any, expected_titles: set[str]) -> tuple[float, list[str]]:
    """Score only expected evidence that is eligible for admission with zero floors.

    Calibration runs with both relevance floors forced to zero. Therefore ``admitted``
    represents the evidence that survives candidate retrieval, resolver/anchor policy,
    reranking and the configured context-k cap before a relevance threshold is applied.
    A high score on the wrong document must never count as a grounded true positive.
    """
    matched_titles: list[str] = []
    scores: list[float] = []
    for item in result.admitted:
        if item.chunk.title not in expected_titles:
            continue
        matched_titles.append(item.chunk.title)
        raw_score = result.reranker_scores.get(item.chunk.id)
        if raw_score is not None:
            scores.append(float(raw_score))
    return max(scores, default=0.0), sorted(set(matched_titles))


def _require_domain_class_separation(observations: list[Observation]) -> None:
    in_domain_unanswerable = [
        item.domain_score
        for item in observations
        if item.case_class == "in_domain_unanswerable"
    ]
    out_of_domain = [
        item.domain_score for item in observations if item.case_class == "out_of_domain"
    ]
    if not in_domain_unanswerable or not out_of_domain:
        raise ValueError(
            "calibration requires both in_domain_unanswerable and out_of_domain observations"
        )
    if min(in_domain_unanswerable) <= max(out_of_domain):
        raise ValueError(
            "domain relevance calibration overlap: in_domain_unanswerable and out_of_domain "
            "cannot be separated by one floor"
        )


def _best_threshold(samples: list[tuple[float, bool]]) -> tuple[float, int, int, int, int]:
    positives = [score for score, expected in samples if expected]
    if not positives:
        raise ValueError("calibration requires positive observations")
    thresholds = sorted({0.0, 1.0, *positives, *(score for score, _expected in samples)})
    ranked: list[tuple[int, int, float, int, int, int, int]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for score, expected in samples:
            predicted = score >= threshold
            if expected and predicted:
                tp += 1
            elif expected:
                fn += 1
            elif predicted:
                fp += 1
            else:
                tn += 1
        ranked.append((fp, fp + fn, -threshold, tp, fp, tn, fn))
    _fp_rank, _error_rank, negative_threshold, tp, fp, tn, fn = min(ranked)
    return -negative_threshold, tp, fp, tn, fn


def _recall(tp: int, fn: int) -> float:
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    # Measure the backend before applying any admission threshold. This is labelled,
    # offline calibration; it is not online self-calibration from user traffic.
    os.environ["RETRIEVAL_DOMAIN_MIN_RELEVANCE"] = "0.0"
    os.environ["RETRIEVAL_MIN_RELEVANCE"] = "0.0"

    from rag_ops_guard.app import knowledge_search
    from rag_ops_guard.domain.models import QueryContext

    observations: list[Observation] = []
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    for case in cases:
        result = knowledge_search().search(case["query"], QueryContext(), query_mode="probe")
        case_class = str(case["class"])
        expected_titles = {str(title) for title in case.get("expected_titles", [])}
        matched_titles: list[str] = []
        if case_class == "grounded":
            grounded_score, matched_titles = expected_grounded_score(result, expected_titles)
        else:
            grounded_score = float(result.grounded_relevance)

        observation = Observation(
            case_id=case["id"],
            case_class=case_class,
            domain_score=float(result.domain_relevance),
            grounded_score=grounded_score,
        )
        observations.append(observation)
        expected_note = (
            f" expected_matched={matched_titles}" if case_class == "grounded" else ""
        )
        print(
            f"{observation.case_id}: class={observation.case_class} "
            f"domain={observation.domain_score:.6f} "
            f"grounded={observation.grounded_score:.6f}{expected_note}"
        )

    try:
        floors = calibrate_floors(observations)
    except ValueError as exc:
        raise SystemExit(f"relevance calibration failed: {exc}") from exc

    if (
        floors.domain_false_positives != 0
        or floors.grounded_false_positives != 0
        or floors.domain_recall < 0.80
        or floors.grounded_recall < 0.80
    ):
        raise SystemExit(
            "relevance calibration failed: require zero false positives and >=80% recall "
            "for both domain and grounded signals"
        )

    print(
        "calibrated relevance floors: "
        f"domain={floors.domain_floor:.6f} grounded={floors.grounded_floor:.6f} "
        f"domain_recall={floors.domain_recall:.3f} "
        f"grounded_recall={floors.grounded_recall:.3f}"
    )
    output = (
        f"export RETRIEVAL_DOMAIN_MIN_RELEVANCE={floors.domain_floor:.6f}\n"
        f"export RETRIEVAL_MIN_RELEVANCE={floors.grounded_floor:.6f}\n"
    )
    if args.env_file:
        args.env_file.parent.mkdir(parents=True, exist_ok=True)
        args.env_file.write_text(output, encoding="utf-8")
        print(f"wrote calibrated relevance environment: {args.env_file}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
