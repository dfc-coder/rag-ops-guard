from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/datasets/retrieval-calibration-v1.json"


@dataclass(frozen=True)
class Observation:
    case_id: str
    positive: bool
    score: float


def _best_threshold(observations: list[Observation]) -> tuple[float, int, int, int, int]:
    thresholds = sorted({0.0, 1.0, *(item.score for item in observations)})
    candidates: list[tuple[int, int, float, int, int, int, int]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for item in observations:
            predicted = item.score >= threshold
            if item.positive and predicted:
                tp += 1
            elif item.positive:
                fn += 1
            elif predicted:
                fp += 1
            else:
                tn += 1
        errors = fp + fn
        # Prefer zero false positives; then fewer total errors; then the higher floor.
        candidates.append((fp, errors, -threshold, tp, fp, tn, fn))
    fp, _errors, neg_threshold, tp, fp, tn, fn = min(candidates)
    return -neg_threshold, tp, fp, tn, fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    # Observe raw top scores without pre-filtering admission.
    os.environ["RETRIEVAL_MIN_RELEVANCE"] = "0.0"
    from rag_ops_guard.app import knowledge_search
    from rag_ops_guard.domain.models import QueryContext

    observations: list[Observation] = []
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    for case in cases:
        result = knowledge_search().search(case["query"], QueryContext())
        score = float(result.relevance)
        positive = case["label"] == "positive"
        observations.append(Observation(case["id"], positive, score))
        print(f"{case['id']}: label={case['label']} top_score={score:.6f}")

    threshold, tp, fp, tn, fn = _best_threshold(observations)
    positives = max(1, tp + fn)
    negatives = max(1, tn + fp)
    recall = tp / positives
    specificity = tn / negatives
    print(
        "calibrated retrieval floor: "
        f"threshold={threshold:.6f} tp={tp} fp={fp} tn={tn} fn={fn} "
        f"recall={recall:.3f} specificity={specificity:.3f}"
    )

    if fp != 0 or recall < 0.80:
        raise SystemExit(
            "retrieval admission calibration failed: require zero negative admissions and >=80% positive recall"
        )

    line = f"export RETRIEVAL_MIN_RELEVANCE={threshold:.6f}\n"
    if args.env_file:
        args.env_file.write_text(line, encoding="utf-8")
    else:
        print(line, end="")


if __name__ == "__main__":
    main()
