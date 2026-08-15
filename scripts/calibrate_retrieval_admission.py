from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/datasets/retrieval-calibration-v2.json"


@dataclass(frozen=True)
class Observation:
    case_id: str
    case_class: str
    score: float

    @property
    def should_admit(self) -> bool:
        return self.case_class == "grounded"


def _best_threshold(observations: list[Observation]) -> tuple[float, int, int, int, int]:
    thresholds = sorted({0.0, 1.0, *(item.score for item in observations)})
    candidates: list[tuple[int, int, float, int, int, int, int]] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for item in observations:
            predicted = item.score >= threshold
            if item.should_admit and predicted:
                tp += 1
            elif item.should_admit:
                fn += 1
            elif predicted:
                fp += 1
            else:
                tn += 1
        errors = fp + fn
        candidates.append((fp, errors, -threshold, tp, fp, tn, fn))
    _fp_rank, _errors, neg_threshold, tp, fp, tn, fn = min(candidates)
    return -neg_threshold, tp, fp, tn, fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    # Observe the learned reranker score before the production admission floor is applied.
    os.environ["RETRIEVAL_MIN_RELEVANCE"] = "0.0"
    from rag_ops_guard.app import knowledge_search
    from rag_ops_guard.domain.models import QueryContext

    observations: list[Observation] = []
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    for case in cases:
        result = knowledge_search().search(case["query"], QueryContext())
        score = float(result.relevance)
        observation = Observation(case["id"], case["class"], score)
        observations.append(observation)
        print(f"{case['id']}: class={case['class']} top_score={score:.6f}")

    threshold, tp, fp, tn, fn = _best_threshold(observations)
    positives = max(1, tp + fn)
    negatives = max(1, tn + fp)
    recall = tp / positives
    specificity = tn / negatives
    print(
        "calibrated retrieval admission floor: "
        f"threshold={threshold:.6f} tp={tp} fp={fp} tn={tn} fn={fn} "
        f"recall={recall:.3f} specificity={specificity:.3f}"
    )
    print(
        "note: in_domain_unanswerable and out_of_domain are intentionally both non-admissible "
        "for retrieval; the ConversationAgent distinguishes them by tool choice, not by this floor."
    )

    if fp != 0 or recall < 0.80:
        raise SystemExit(
            "retrieval admission calibration failed: require zero non-grounded admissions "
            "and >=80% grounded recall"
        )

    line = f"export RETRIEVAL_MIN_RELEVANCE={threshold:.6f}\n"
    if args.env_file:
        args.env_file.write_text(line, encoding="utf-8")
    else:
        print(line, end="")


if __name__ == "__main__":
    main()
