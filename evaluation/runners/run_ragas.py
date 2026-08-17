from __future__ import annotations

import json
import math
import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
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
from ragas.run_config import RunConfig

from rag_ops_guard.evaluation.gates import (
    JudgePolicy,
    enforce_metric_thresholds,
    grounded_segment_text,
    require_calibration_policy,
)
from rag_ops_guard.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeIdentity,
    judge_connection_from_env,
)

JUDGE_POLICY_PATH = Path("artifacts/evaluation/judge-policy.json")
GOLDEN_SAMPLES_PATH = Path("artifacts/evaluation/golden-samples.json")


@dataclass(frozen=True)
class RagasSample:
    case_id: str
    user_input: str
    retrieved_contexts: list[str]
    grounded_response: str
    reference: str


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


def _collect_samples(path: Path = GOLDEN_SAMPLES_PATH) -> list[RagasSample]:
    """Collect only the document-grounded portion of Golden answers for RAGAS.

    Golden owns direct/ungrounded-answer correctness, status, safety and citation contracts.
    RAGAS owns the quality of actual RAG output, so context/reference metrics are evaluated
    only when the response contains grounded segments and a reference answer exists.
    """
    if not path.is_file():
        raise SystemExit(
            f"Golden samples not found: {path}. Run evaluation/runners/run_golden.py first."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("golden-samples.json must contain a JSON list")

    s3 = _s3()
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    samples: list[RagasSample] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        expected_status = str(row.get("expected_status") or "")
        if not expected_status.startswith("answered_"):
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise SystemExit(f"golden sample {row.get('case_id')} has no response payload")
        actual_status = str(payload.get("status") or "")
        if not actual_status.startswith("answered_"):
            raise SystemExit(
                f"RAGAS expected an answered Golden response for {row.get('case_id')}, "
                f"got {actual_status or '<missing>'}"
            )

        grounded_response = grounded_segment_text(payload)
        if not grounded_response:
            continue

        reference = str(row.get("reference_answer") or "").strip()
        if not reference:
            raise SystemExit(
                f"grounded golden sample {row.get('case_id')} is missing reference_answer"
            )
        contexts = [
            _context_text(s3, bucket, str(citation["s3_key"]))
            for citation in payload.get("citations", [])
            if isinstance(citation, dict) and citation.get("s3_key")
        ]
        if not contexts:
            raise SystemExit(
                f"grounded golden sample {row.get('case_id')} has no retrievable citation contexts"
            )
        samples.append(
            RagasSample(
                case_id=str(row.get("case_id") or ""),
                user_input=str(row.get("ragas_question") or row.get("question") or ""),
                retrieved_contexts=contexts,
                grounded_response=grounded_response,
                reference=reference,
            )
        )
    if not samples:
        raise SystemExit("Golden samples contain no grounded/reference cases for RAGAS")
    return samples


def _dataset(samples: list[RagasSample]) -> EvaluationDataset:
    return EvaluationDataset.from_list(
        [
            {
                "user_input": sample.user_input,
                "retrieved_contexts": sample.retrieved_contexts,
                "response": sample.grounded_response,
                "reference": sample.reference,
            }
            for sample in samples
        ]
    )


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
    if not gating_enabled:
        return
    failures: list[str] = []
    for metric, mean_floor in means.items():
        try:
            enforce_metric_thresholds(
                metric,
                metric_values[metric],
                mean_floor=float(mean_floor),
                per_case_floor=(float(per_case[metric]) if metric in per_case else None),
            )
        except ValueError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("RAGAS thresholds failed:\n" + "\n".join(failures))


def _calibration_required() -> bool:
    value = os.environ.get("RAGAS_REQUIRE_CALIBRATION", "1").strip()
    if value not in {"0", "1"}:
        raise SystemExit("RAGAS_REQUIRE_CALIBRATION must be '0' or '1'")
    return value == "1"


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if value < 1:
        raise SystemExit(f"{name} must be >= 1")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be > 0")
    return value


@contextmanager
def suite_wall_timeout(seconds: float) -> Iterator[None]:
    """Bound the whole local evaluation, independent of library-level retries/timeouts."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def on_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"RAGAS suite exceeded hard wall timeout of {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, on_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _load_judge_policy(identity: JudgeIdentity) -> JudgePolicy | None:
    if not JUDGE_POLICY_PATH.exists():
        return None
    payload = json.loads(JUDGE_POLICY_PATH.read_text(encoding="utf-8"))
    expected = {
        "judge_provider": identity.provider,
        "judge_model": identity.model,
        "judge_prompt_sha256": identity.prompt_sha256,
        "evaluation_dataset_sha256": identity.dataset_sha256,
    }
    mismatches = [
        f"{key}: policy={payload.get(key)!r}, active={value!r}"
        for key, value in expected.items()
        if str(payload.get(key) or "") != value
    ]
    if mismatches:
        raise SystemExit(
            "judge calibration is stale for the active evaluator; recalibrate. " + "; ".join(mismatches)
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
    judge = judge_connection_from_env()
    judge_timeout = _positive_float_env("RAGAS_JUDGE_TIMEOUT_SECONDS", 120.0)
    embedding_timeout = _positive_float_env("EMBEDDING_TIMEOUT_SECONDS", 60.0)
    suite_timeout = _positive_float_env("RAGAS_SUITE_TIMEOUT_SECONDS", 1800.0)
    run_config = RunConfig(
        timeout=_positive_int_env("RAGAS_OPERATION_TIMEOUT_SECONDS", 120),
        max_retries=_positive_int_env("RAGAS_MAX_RETRIES", 2),
        max_wait=_positive_int_env("RAGAS_MAX_WAIT_SECONDS", 5),
        max_workers=_positive_int_env("RAGAS_MAX_WORKERS", 2),
        seed=42,
    )

    print(
        f"RAGAS START: judge={judge.identity.provider}/{judge.identity.model}; "
        f"suite-timeout={suite_timeout:g}s; reusing artifacts/evaluation/golden-samples.json",
        flush=True,
    )
    llm_client = AsyncOpenAI(
        base_url=judge.base_url,
        api_key=judge.api_key,
        timeout=judge_timeout,
        max_retries=0,
    )
    embedding_client = AsyncOpenAI(
        base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
        api_key="local",
        timeout=embedding_timeout,
        max_retries=0,
    )
    evaluator_llm = llm_factory(
        judge.identity.model,
        client=llm_client,
        temperature=0.0,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        client=embedding_client,
        model=os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
    )

    samples = _collect_samples()
    print(
        f"RAGAS DATASET: {len(samples)} grounded/reference case(s); "
        f"workers={run_config.max_workers} operation-timeout={run_config.timeout}s",
        flush=True,
    )
    try:
        with suite_wall_timeout(suite_timeout):
            result = evaluate(
                dataset=_dataset(samples),
                metrics=[
                    Faithfulness(),
                    LLMContextPrecisionWithReference(),
                    LLMContextRecall(),
                    ResponseRelevancy(),
                ],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=True,
                show_progress=True,
            )
    except TimeoutError as exc:
        raise SystemExit(str(exc)) from exc

    frame = result.to_pandas()
    case_ids = [sample.case_id for sample in samples]
    metric_values = {
        "faithfulness": _metric_values(frame, "faithfulness", case_ids),
        "context_precision": _metric_values(
            frame,
            "llm_context_precision_with_reference",
            case_ids,
        ),
        "context_recall": _metric_values(frame, "context_recall", case_ids),
        "response_relevancy": _metric_values(frame, "answer_relevancy", case_ids),
    }
    summary = {
        name: sum(values.values()) / len(values) if values else 0.0
        for name, values in metric_values.items()
    }
    summary.update(
        {
            "evaluator_provider": judge.identity.provider,
            "evaluator": judge.identity.model,
            "judge_prompt_sha256": judge.identity.prompt_sha256,
            "evaluation_dataset_sha256": judge.identity.dataset_sha256,
            "embedding_evaluator": os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
            "samples": len(samples),
            "scope": "grounded_segments_with_reference",
            "run_config": {
                "timeout": run_config.timeout,
                "max_retries": run_config.max_retries,
                "max_wait": run_config.max_wait,
                "max_workers": run_config.max_workers,
                "seed": run_config.seed,
                "suite_timeout": suite_timeout,
            },
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

    policy = _load_judge_policy(judge.identity)
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
    print(json.dumps(summary, indent=2), flush=True)

    try:
        policy = require_calibration_policy(policy, required=_calibration_required())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if policy is None:
        print(
            "RAGAS calibration absent; metrics are informational because "
            "RAGAS_REQUIRE_CALIBRATION=0 was set explicitly.",
            flush=True,
        )
        return

    means = {name: float(value) for name, value in thresholds["ragas"].items()}
    if policy.mean_floor is not None:
        means["faithfulness"] = policy.mean_floor
    enforce_thresholds(
        metric_values,
        means=means,
        per_case={name: float(value) for name, value in thresholds["ragas_per_case"].items()},
        gating_enabled=policy.gating_enabled,
    )

    if not policy.gating_enabled:
        print(
            "RAGAS judge agreement is <= 6/10; metrics are informational. "
            "Deterministic citation_validity and segment_integrity remain blocking.",
            flush=True,
        )


if __name__ == "__main__":
    main()
