from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.runners import run_ragas


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
            "question": "How many retries?",
            "expected_status": "answered_grounded",
            "reference_answer": "Three retries are allowed.",
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
        },
    ]
    path = tmp_path / "golden-samples.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(run_ragas, "_s3", lambda: object())
    monkeypatch.setattr(run_ragas, "_context_text", lambda _s3, _bucket, _key: "Policy context")

    samples = run_ragas._collect_samples(path)

    assert [sample.case_id for sample in samples] == ["grounded"]
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
