from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from evaluation.runners import run_ragas
from rag_ops_guard.evaluation.judge import JudgeIdentity, evaluation_dataset_sha256


class _Column:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _Frame:
    def __init__(self, values: dict[str, list[float]]) -> None:
        self._values = values

    def __getitem__(self, key: str) -> _Column:
        return _Column(self._values[key])


def test_ragas_collects_grounded_reference_cases_and_skips_ungrounded_without_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "case_id": "ungrounded",
            "question": "What is missing?",
            "expected_status": "answered_ungrounded",
            "reference_answer": "",
            "payload": {
                "status": "answered_ungrounded",
                "segments": [{"text": "No grounded evidence is available.", "citations": []}],
                "citations": [],
            },
        },
        {
            "case_id": "grounded",
            "question": "How many retries, and give a generic Python example?",
            "ragas_question": "How many retries?",
            "expected_status": "answered_mixed",
            "reference_answer": "Three retries are allowed.",
            "payload": {
                "status": "answered_mixed",
                "segments": [
                    {
                        "text": "Three retries are allowed.",
                        "citations": [{"chunk_id": "c1", "s3_key": "chunks/c1.json"}],
                    },
                    {"text": "Example: sleep(1)", "citations": []},
                ],
                "citations": [{"chunk_id": "c1", "s3_key": "chunks/c1.json"}],
            },
        },
    ]
    path = tmp_path / "golden-samples.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(run_ragas, "_s3", lambda: object())
    monkeypatch.setattr(run_ragas, "_context_text", lambda _s3, _bucket, _key: "Policy context")

    samples = run_ragas._collect_samples(path)

    assert [sample.case_id for sample in samples] == ["grounded"]
    assert samples[0].user_input == "How many retries?"
    assert samples[0].reference == "Three retries are allowed."
    assert samples[0].retrieved_contexts == ["Policy context"]


def test_ragas_rejects_grounded_case_without_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "case_id": "grounded",
            "question": "How many retries?",
            "expected_status": "answered_grounded",
            "reference_answer": "",
            "payload": {
                "status": "answered_grounded",
                "segments": [
                    {
                        "text": "Three retries are allowed.",
                        "citations": [{"chunk_id": "c1", "s3_key": "chunks/c1.json"}],
                    }
                ],
                "citations": [{"chunk_id": "c1", "s3_key": "chunks/c1.json"}],
            },
        }
    ]
    path = tmp_path / "golden-samples.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(run_ragas, "_s3", lambda: object())

    with pytest.raises(SystemExit, match="missing reference_answer"):
        run_ragas._collect_samples(path)


def test_ragas_run_config_environment_must_be_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGAS_MAX_WORKERS", "0")

    with pytest.raises(SystemExit, match="RAGAS_MAX_WORKERS must be >= 1"):
        run_ragas._positive_int_env("RAGAS_MAX_WORKERS", 2)


def test_ragas_suite_hard_timeout_bounds_complete_evaluation() -> None:
    if not hasattr(run_ragas.signal, "setitimer"):
        pytest.skip("hard wall timeout requires POSIX setitimer")

    with pytest.raises(TimeoutError, match="RAGAS suite exceeded"):
        with run_ragas.suite_wall_timeout(0.02):
            time.sleep(0.10)


def test_repository_golden_grounded_cases_have_references() -> None:
    rows: list[dict[str, object]] = []
    for path in (
        Path("evaluation/datasets/golden-v1.json"),
        Path("evaluation/datasets/golden-mixed-v1.json"),
    ):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    missing = [
        str(row["id"])
        for row in rows
        if str(row.get("expected_status")) in {"answered_grounded", "answered_mixed"}
        and row.get("expected_source_ids")
        and not str(row.get("reference_answer") or "").strip()
    ]

    assert missing == []


def test_pinned_ragas_metric_names_match_runner_result_columns() -> None:
    metric_names = [
        Faithfulness().name,
        LLMContextPrecisionWithReference().name,
        LLMContextRecall().name,
        ResponseRelevancy().name,
    ]
    assert metric_names == [
        "faithfulness",
        "llm_context_precision_with_reference",
        "context_recall",
        "answer_relevancy",
    ]

    frame = _Frame(
        {
            "faithfulness": [1.0],
            "llm_context_precision_with_reference": [0.9],
            "context_recall": [0.8],
            "answer_relevancy": [0.7],
        }
    )
    assert run_ragas._metric_values(frame, "faithfulness", ["one"]) == {"one": 1.0}
    assert run_ragas._metric_values(frame, "answer_relevancy", ["one"]) == {"one": 0.7}


def test_judge_dataset_identity_covers_base_and_mixed_suites(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    mixed = tmp_path / "mixed.json"
    base.write_text('[{"id":"base"}]', encoding="utf-8")
    mixed.write_text('[{"id":"mixed"}]', encoding="utf-8")
    first = evaluation_dataset_sha256((base, mixed))

    mixed.write_text('[{"id":"mixed-changed"}]', encoding="utf-8")
    second = evaluation_dataset_sha256((base, mixed))

    assert first != second


def test_stale_judge_policy_is_ignored_for_measurement_but_blocks_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "judge-policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "judge_provider": "local",
                "judge_model": "old-model",
                "judge_prompt_sha256": "old-prompt",
                "evaluation_dataset_sha256": "old-dataset",
                "agreement": 1.0,
                "gating_enabled": True,
                "mean_floor": 0.85,
                "calibrated_cutoff": None,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_ragas, "JUDGE_POLICY_PATH", policy_path)
    identity = JudgeIdentity(
        provider="external",
        model="new-model",
        prompt_sha256="new-prompt",
        dataset_sha256="new-dataset",
    )

    assert run_ragas._load_judge_policy(identity, required=False) is None
    with pytest.raises(SystemExit, match="judge calibration is stale"):
        run_ragas._load_judge_policy(identity, required=True)
