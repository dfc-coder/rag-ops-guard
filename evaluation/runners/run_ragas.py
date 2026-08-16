from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import httpx
import yaml
from botocore.config import Config
from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from rag_ops_guard.evaluation.gates import (
    JudgePolicy,
    enforce_metric_thresholds,
    grounded_segment_text,
    require_qwen3_4b_judge,
)

JUDGE_POLICY_PATH = Path("artifacts/evaluation/judge-policy.json")


@dataclass(frozen=True)
class RagasSample:
    case_id: str
    user_input: str
    retrieved_contexts: list[str]
    response: str
    grounded_response: str
    reference: str


def _api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


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


def _is_answer_case(case: dict[str, Any]) -> bool:
    status = str(case.get("expected_status") or "")
    return status == "answered" or status.startswith("answered_")


def _collect_samples() -> list[RagasSample]:
    cases = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    api = _api_url()
    s3 = _s3()
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    samples: list[RagasSample] = []

    for case in cases:
        if not _is_answer_case(case):
            continue
        response = httpx.post(
            f"{api}/v1/query",
            json={"question": case["question"], "context": case.get("context", {})},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        actual_status = str(payload.get("status") or "")
        if not actual_status.startswith("answered"):
            raise SystemExit(
                f"RAGAS collection expected an answered response for {case['id']}, "
                f"got {actual_status or '<missing>'}"
            )
        contexts = [
            _context_text(s3, bucket, str(citation["s3_key"]))
            for citation in payload.get("citations", [])
        ]
        samples.append(
            RagasSample(
                case_id=str(case["id"]),
                user_input=str(case["question"]),
                retrieved_contexts=contexts,
                response=str(payload.get("answer") or ""),
                grounded_response=grounded_segment_text(payload),
                reference=str(case["reference_answer"]),
            )
        )
    return samples


def _dataset(samples: list[RagasSample], *, grounded_only: bool) -> EvaluationDataset:
    records = []
    for sample in samples:
        response = sample.grounded_response if grounded_only else sample.response
        if grounded_only and not response:
            continue
        records.append(
            {
                "user_input": sample.user_input,
                "retrieved_contexts": sample.retrieved_contexts,
                "response": response,
                "reference": sample.reference,
            }
        )
    return EvaluationDataset.from_list(records)


def _metric_values(frame: Any, column: str, case_ids: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for case_id, raw in zip(case_ids, frame[column].tolist(), strict=True):
        score = float(raw)
        if not math.isnan(score):
            values[case_id] = score
    return values


def enforce_thresholds(
    metric_values: dict[str, dict[str, float]],
    *,
    means: dict[str, float],
    per_case: dict[str, float],
    gating_enabled: bool,
) -> None:
    """SPEC-5.1: enforce both aggregate and catastrophic-case floors when RAGAS may gate."""
    if not gating_enabled:
        return
    failures: list[str] = []
    for metric, mean_floor in means.items():
        try:
            enforce_metric_thresholds(
                metric,
                metric_values[metric],
                mean_floor=float(mean_floor),
                per_case_floor=(
                    float(per_case[metric]) if metric in per_case else None
                ),
            )
        except ValueError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("RAGAS thresholds failed:\n" + "\n".join(failures))


def _runtime_judge_model() -> str:
    runtime = os.environ.get("LLM_MODEL", "qwen35-2b-rag")
    judge = os.environ.get("RAGAS_JUDGE_MODEL", runtime)
    if judge != runtime:
        raise SystemExit(
            "SPEC-5.3 forbids a second judge model: RAGAS_JUDGE_MODEL must equal LLM_MODEL"
        )
    try:
        require_qwen3_4b_judge(judge)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return judge


def _load_judge_policy(model: str) -> JudgePolicy | None:
    if not JUDGE_POLICY_PATH.exists():
        return None
    payload = json.loads(JUDGE_POLICY_PATH.read_text(encoding="utf-8"))
    if str(payload.get("model")) != model:
        raise SystemExit(
            "judge calibration model does not match runtime model; recalibrate on the active Qwen3-4B"
        )
    return JudgePolicy(
        agreement=float(payload["agreement"]),
        gating_enabled=bool(payload["gating_enabled"]),
        mean_floor=(
            float(payload["mean_floor"]) if payload.get("mean_floor") is not None else None
        ),
        calibrated_cutoff=(
            float(payload["calibrated_cutoff"])
            if payload.get("calibrated_cutoff") is not None
            else None
        ),
    )


def main() -> None:
    thresholds = yaml.safe_load(Path("evaluation/thresholds.yaml").read_text())
    model = _runtime_judge_model()

    llm_client = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="local",
    )
    embedding_client = AsyncOpenAI(
        base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
        api_key="local",
    )
    evaluator_llm = llm_factory(
        model,
        client=llm_client,
        temperature=0.0,
        system_prompt=(
            "Judge only the provided question, response, reference, and contexts. "
            "Do not use external knowledge. /no_think"
        ),
    )
    evaluator_embeddings = OpenAIEmbeddings(
        client=embedding_client,
        model=os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
    )

    samples = _collect_samples()
    full_result = evaluate(
        dataset=_dataset(samples, grounded_only=False),
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ResponseRelevancy(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    full_frame = full_result.to_pandas()

    grounded_samples = [sample for sample in samples if sample.grounded_response]
    faithfulness_result = evaluate(
        dataset=_dataset(samples, grounded_only=True),
        metrics=[Faithfulness()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    faithfulness_frame = faithfulness_result.to_pandas()

    metric_values = {
        "faithfulness": _metric_values(
            faithfulness_frame,
            "faithfulness",
            [sample.case_id for sample in grounded_samples],
        ),
        "context_precision": _metric_values(
            full_frame,
            "llm_context_precision_with_reference",
            [sample.case_id for sample in samples],
        ),
        "context_recall": _metric_values(
            full_frame,
            "context_recall",
            [sample.case_id for sample in samples],
        ),
        "response_relevancy": _metric_values(
            full_frame,
            "answer_relevancy",
            [sample.case_id for sample in samples],
        ),
    }
    summary = {
        name: sum(values.values()) / len(values) if values else 0.0
        for name, values in metric_values.items()
    }
    summary.update(
        {
            "evaluator": model,
            "embedding_evaluator": os.environ.get(
                "EMBEDDING_MODEL", "qwen3-embedding-0.6b"
            ),
            "samples": len(samples),
            "faithfulness_samples": len(grounded_samples),
        }
    )

    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        row: dict[str, object] = {"case_id": sample.case_id}
        for metric, values in metric_values.items():
            row[metric] = values.get(sample.case_id)
        rows.append(row)
    (output / "ragas-results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    policy = _load_judge_policy(model)
    summary["judge_calibration"] = (
        None
        if policy is None
        else {
            "agreement": policy.agreement,
            "gating_enabled": policy.gating_enabled,
            "mean_floor": policy.mean_floor,
            "calibrated_cutoff": policy.calibrated_cutoff,
        }
    )
    (output / "ragas.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if policy is None:
        raise SystemExit(
            "SPEC-5.3 calibration required: label exactly 10 cases and run "
            "scripts/calibrate_ragas_judge.py against artifacts/evaluation/ragas-results.json"
        )

    means = {name: float(value) for name, value in thresholds["ragas"].items()}
    if policy.mean_floor is not None:
        means["faithfulness"] = policy.mean_floor
    enforce_thresholds(
        metric_values,
        means=means,
        per_case={
            name: float(value) for name, value in thresholds["ragas_per_case"].items()
        },
        gating_enabled=policy.gating_enabled,
    )

    if not policy.gating_enabled:
        print(
            "RAGAS judge agreement is <= 6/10; metrics are informational. "
            "Deterministic citation/segment/security gates remain authoritative."
        )


if __name__ == "__main__":
    main()
