from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ops_guard.retrieval.calibration import (
    corpus_fingerprint,
    derive_separating_threshold,
    file_fingerprint,
    load_calibrated_threshold,
)


def test_derives_threshold_between_labeled_populations() -> None:
    result = derive_separating_threshold(
        positive_scores=[0.14, 0.31, 0.82],
        negative_scores=[0.00002, 0.004, 0.03],
    )

    assert result.max_negative == 0.03
    assert result.min_positive == 0.14
    assert result.threshold == pytest.approx(0.085)
    assert result.separation == pytest.approx(0.11)


def test_refuses_to_invent_threshold_when_scores_overlap() -> None:
    with pytest.raises(ValueError, match="not separable"):
        derive_separating_threshold(
            positive_scores=[0.10, 0.20],
            negative_scores=[0.02, 0.12],
        )


def test_load_requires_matching_model_corpus_and_dataset(tmp_path: Path) -> None:
    corpus = tmp_path / "knowledge-base"
    corpus.mkdir()
    (corpus / "doc.md").write_text("# Current knowledge\n", encoding="utf-8")
    dataset = tmp_path / "calibration-dataset.json"
    dataset.write_text('[{"label":"positive"}]', encoding="utf-8")
    artifact = tmp_path / "calibration.json"
    artifact.write_text(
        json.dumps(
            {
                "reranker_model": "bge-reranker-v2-m3",
                "corpus_fingerprint": corpus_fingerprint(corpus),
                "dataset_fingerprint": file_fingerprint(dataset),
                "threshold": 0.085,
            }
        ),
        encoding="utf-8",
    )

    assert load_calibrated_threshold(
        artifact,
        reranker_model="bge-reranker-v2-m3",
        corpus_root=corpus,
        dataset_path=dataset,
    ) == pytest.approx(0.085)

    with pytest.raises(RuntimeError, match="model does not match"):
        load_calibrated_threshold(
            artifact,
            reranker_model="other-model",
            corpus_root=corpus,
            dataset_path=dataset,
        )

    (corpus / "doc.md").write_text("# Changed knowledge\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corpus fingerprint is stale"):
        load_calibrated_threshold(
            artifact,
            reranker_model="bge-reranker-v2-m3",
            corpus_root=corpus,
            dataset_path=dataset,
        )

    (corpus / "doc.md").write_text("# Current knowledge\n", encoding="utf-8")
    dataset.write_text('[{"label":"negative"}]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="dataset fingerprint is stale"):
        load_calibrated_threshold(
            artifact,
            reranker_model="bge-reranker-v2-m3",
            corpus_root=corpus,
            dataset_path=dataset,
        )
