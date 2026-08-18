from __future__ import annotations

from scripts import calibrate_relevance_floors
from scripts.local import ops, provision


def test_calibration_values_are_publishable_config_overrides() -> None:
    floors = calibrate_relevance_floors.CalibrationFloors(
        domain_floor=0.31,
        grounded_floor=0.59,
        domain_false_positives=0,
        grounded_false_positives=0,
        domain_recall=0.9,
        grounded_recall=1.0,
    )

    assert calibrate_relevance_floors.calibration_values(floors) == {
        "retrieval_domain_min_relevance": 0.31,
        "retrieval_min_relevance": 0.59,
    }


def test_lambda_environment_contains_bootstrap_not_runtime_tuning() -> None:
    environment = provision.lambda_environment(config_hash="b" * 64)

    assert environment["CONFIG_SOURCE"] == "db"
    assert environment["CONFIG_HASH"] == "b" * 64
    assert "CONFIG_TABLE" in environment
    assert "RETRIEVAL_TOP_K" not in environment
    assert "RETRIEVAL_CONTEXT_K" not in environment
    assert "RETRIEVAL_DOMAIN_MIN_RELEVANCE" not in environment
    assert "RETRIEVAL_MIN_RELEVANCE" not in environment
    assert "CHUNK_TOKENS" not in environment
    assert "LLM_TEMPERATURE" not in environment


def test_status_hash_contract_detects_divergence() -> None:
    assert ops._config_hash_matches("a" * 64, {"CONFIG_HASH": "a" * 64})
    assert not ops._config_hash_matches("a" * 64, {"CONFIG_HASH": "b" * 64})
    assert not ops._config_hash_matches("a" * 64, {})
