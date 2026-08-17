from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.calibrate_relevance_floors import (
    Observation,
    calibrate_floors,
    expected_grounded_score,
    serialize_floor,
)


def test_calibration_allows_overlap_when_zero_fp_threshold_preserves_recall() -> None:
    observations = [
        Observation("g1", "grounded", domain_score=0.99, grounded_score=0.95),
        Observation("g2", "grounded", domain_score=0.91, grounded_score=0.88),
        Observation("iu-high", "in_domain_unanswerable", domain_score=0.70, grounded_score=0.10),
        Observation("iu-mid", "in_domain_unanswerable", domain_score=0.40, grounded_score=0.08),
        Observation("iu-low", "in_domain_unanswerable", domain_score=0.12, grounded_score=0.00),
        Observation("ood-high", "out_of_domain", domain_score=0.19, grounded_score=0.05),
        Observation("ood-low", "out_of_domain", domain_score=0.08, grounded_score=0.02),
    ]

    floors = calibrate_floors(observations)

    assert floors.domain_floor == pytest.approx(0.295)
    assert floors.domain_false_positives == 0
    assert floors.domain_recall == 0.8
    assert floors.grounded_false_positives == 0
    assert floors.grounded_recall == 1.0


def test_calibration_returns_separate_domain_and_grounded_floors() -> None:
    observations = [
        Observation("g1", "grounded", domain_score=0.91, grounded_score=0.89),
        Observation("g2", "grounded", domain_score=0.84, grounded_score=0.80),
        Observation("iu1", "in_domain_unanswerable", domain_score=0.71, grounded_score=0.22),
        Observation("iu2", "in_domain_unanswerable", domain_score=0.66, grounded_score=0.18),
        Observation("ood1", "out_of_domain", domain_score=0.25, grounded_score=0.12),
        Observation("ood2", "out_of_domain", domain_score=0.18, grounded_score=0.08),
    ]

    floors = calibrate_floors(observations)

    assert 0.25 < floors.domain_floor <= 0.66
    assert 0.22 < floors.grounded_floor <= 0.80
    assert floors.domain_false_positives == 0
    assert floors.grounded_false_positives == 0
    assert floors.domain_recall == 1.0
    assert floors.grounded_recall == 1.0


def test_domain_calibration_treats_unanswerable_corpus_queries_as_in_domain() -> None:
    observation = Observation(
        "iu",
        "in_domain_unanswerable",
        domain_score=0.70,
        grounded_score=0.10,
    )

    assert observation.should_be_domain_related is True
    assert observation.should_be_grounded is False


def test_expected_grounded_score_uses_only_expected_admitted_evidence() -> None:
    correct = SimpleNamespace(chunk=SimpleNamespace(title="Payment Retry Policy", id="correct"))
    wrong = SimpleNamespace(chunk=SimpleNamespace(title="Vendor Troubleshooting Note", id="wrong"))
    result = SimpleNamespace(
        admitted=[wrong, correct],
        reranker_scores={"wrong": 0.99, "correct": 0.71},
    )

    score, matched = expected_grounded_score(result, {"Payment Retry Policy"})

    assert score == 0.71
    assert matched == ["Payment Retry Policy"]


def test_expected_grounded_score_is_zero_when_only_wrong_evidence_survives() -> None:
    wrong = SimpleNamespace(chunk=SimpleNamespace(title="Vendor Troubleshooting Note", id="wrong"))
    result = SimpleNamespace(admitted=[wrong], reranker_scores={"wrong": 0.99})

    score, matched = expected_grounded_score(result, {"Payment Retry Policy"})

    assert score == 0.0
    assert matched == []


def test_serialized_floor_round_trips_without_rounding_above_positive_score() -> None:
    measured = 0.9041435012345678

    serialized = serialize_floor(measured)

    assert float(serialized) == measured
    assert float(serialized) <= measured
