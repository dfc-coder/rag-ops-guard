from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

from rag_ops_guard.evaluation.gates import grounded_segment_text

GOLDEN_PATH = Path("artifacts/evaluation/golden-samples.json")
RAGAS_PATH = Path("artifacts/evaluation/ragas-results.json")
OUTPUT_PATH = Path("artifacts/evaluation/judge-human-review.json")
REVIEW_CASES = 10
REVIEW_SEED = 42


def _s3() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        config=Config(s3={"addressing_style": "path"}),
    )


def _context_text(s3: Any, bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return str(payload["text"])


def main() -> None:
    if not GOLDEN_PATH.is_file() or not RAGAS_PATH.is_file():
        raise SystemExit("Golden and RAGAS artifacts are required before human review preparation")

    golden_rows = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    ragas_rows = json.loads(RAGAS_PATH.read_text(encoding="utf-8"))
    golden_by_id = {
        str(row.get("case_id")): row for row in golden_rows if isinstance(row, dict)
    }
    candidates = [
        row
        for row in ragas_rows
        if isinstance(row, dict)
        and row.get("case_id") is not None
        and row.get("faithfulness") is not None
        and str(row.get("case_id")) in golden_by_id
    ]
    if len(candidates) < REVIEW_CASES:
        raise SystemExit(
            f"human calibration requires at least {REVIEW_CASES} scored grounded cases; "
            f"found {len(candidates)}"
        )

    ordered = sorted(candidates, key=lambda row: str(row["case_id"]))
    selected = random.Random(REVIEW_SEED).sample(ordered, REVIEW_CASES)
    selected.sort(key=lambda row: str(row["case_id"]))

    s3 = _s3()
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    review: list[dict[str, object]] = []
    for score_row in selected:
        case_id = str(score_row["case_id"])
        golden = golden_by_id[case_id]
        payload = golden.get("payload")
        if not isinstance(payload, dict):
            raise SystemExit(f"golden sample {case_id} has no response payload")
        contexts = [
            _context_text(s3, bucket, str(citation["s3_key"]))
            for citation in payload.get("citations", [])
            if isinstance(citation, dict) and citation.get("s3_key")
        ]
        review.append(
            {
                "case_id": case_id,
                "question": str(golden.get("ragas_question") or golden.get("question") or ""),
                "grounded_response": grounded_segment_text(payload),
                "evidence": contexts,
                "judge_faithfulness": float(score_row["faithfulness"]),
                "human_pass": None,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"human judge review ready: {OUTPUT_PATH} ({len(review)} cases)")
    print("Fill only human_pass with true or false for every case, then run make eval-judge-calibrate")


if __name__ == "__main__":
    main()
