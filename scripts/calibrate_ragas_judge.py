from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_ops_guard.evaluation.gates import HumanJudgeCase, calibrate_judge_policy
from rag_ops_guard.evaluation.judge import judge_identity_from_env

BASELINE_JUDGE_CUTOFF = 0.85


def _load_human_labels(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(
            f"human calibration labels not found: {path}. "
            "Create exactly 10 entries with case_id and human_pass after manual review."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("human calibration labels must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("evaluation/datasets/judge-human-10.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/evaluation/ragas-results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/judge-policy.json"),
    )
    args = parser.parse_args()

    identity = judge_identity_from_env()
    labels = _load_human_labels(args.labels)
    if not args.results.exists():
        raise SystemExit(
            f"RAGAS results not found: {args.results}. Run evaluation/runners/run_ragas.py first."
        )

    result_rows = json.loads(args.results.read_text(encoding="utf-8"))
    scores = {
        str(row["case_id"]): float(row["faithfulness"])
        for row in result_rows
        if isinstance(row, dict)
        and row.get("case_id") is not None
        and row.get("faithfulness") is not None
    }

    comparisons: list[HumanJudgeCase] = []
    for label in labels:
        case_id = str(label.get("case_id") or "").strip()
        human_pass = label.get("human_pass")
        if not case_id or not isinstance(human_pass, bool):
            raise SystemExit(
                "every human calibration entry requires non-empty case_id and boolean human_pass"
            )
        if case_id not in scores:
            raise SystemExit(f"no faithfulness score found for human-labelled case {case_id}")
        score = scores[case_id]
        comparisons.append(
            HumanJudgeCase(
                case_id=case_id,
                human_pass=human_pass,
                judge_pass=score >= BASELINE_JUDGE_CUTOFF,
                judge_score=score,
            )
        )

    try:
        policy = calibrate_judge_policy(comparisons)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output = {
        "judge_provider": identity.provider,
        "judge_model": identity.model,
        "judge_prompt_sha256": identity.prompt_sha256,
        "evaluation_dataset_sha256": identity.dataset_sha256,
        "cases": [case.case_id for case in comparisons],
        "agreement": policy.agreement,
        "gating_enabled": policy.gating_enabled,
        "mean_floor": policy.mean_floor,
        "calibrated_cutoff": policy.calibrated_cutoff,
        "baseline_cutoff_used_for_agreement": BASELINE_JUDGE_CUTOFF,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))

    if policy.agreement < 0.7:
        print(
            "Judge agreement is <= 6/10. RAGAS is informational and must not gate release; "
            "deterministic citation_validity and segment_integrity remain blocking."
        )
    elif policy.agreement < 0.9:
        print(
            "Judge agreement is 7-8/10. RAGAS may gate with the measured lower mean cutoff plus "
            "per-case floors."
        )
    else:
        print("Judge agreement is 9-10/10. The plan's 0.85 mean floor may gate.")


if __name__ == "__main__":
    main()
