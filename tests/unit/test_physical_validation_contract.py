from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _env_value(name: str) -> str:
    prefix = f"{name}="
    for line in _read(".env.example").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    raise AssertionError(f"{name} missing from .env.example")


def test_physical_runtime_uses_one_generation_model_alias() -> None:
    alias = _env_value("LLM_MODEL")
    compose = _read("docker/docker-compose.yml")
    provision = _read("scripts/local/provision.py")
    release = _read(".github/workflows/release-validation.yml")

    assert f"- {alias}" in compose
    assert '"LLM_MODEL": LLM_MODEL' in provision
    assert "qwen35-2b-rag" not in provision
    assert f"LLM_MODEL: {alias}" in release
    assert f"RAGAS_JUDGE_MODEL: {alias}" in release


def test_physical_floci_version_supports_api_id_override() -> None:
    compose = _read("docker/docker-compose.yml")
    hosted_ci = _read(".github/workflows/ci.yml")

    assert "floci/floci:1.6.0" in compose
    assert "floci/floci:1.6.0" in hosted_ci


def test_physical_smokes_use_segmented_status_contract() -> None:
    smoke = _read("scripts/smoke.py")
    e2e = _read("tests/e2e/test_real_local_beta.py")
    beta_smoke = _read("scripts/beta_freeze_smoke.py")

    assert 'payload["status"] != "answered"' not in smoke
    assert 'payload["status"] == "answered"' not in e2e
    assert "QueryStatus.INSUFFICIENT_EVIDENCE" not in beta_smoke
    assert "QueryStatus.ANSWERED or" not in beta_smoke
    assert "answered_grounded" in smoke
    assert "answered_grounded" in e2e


def test_release_supersedes_obsolete_physical_runs() -> None:
    release = _read(".github/workflows/release-validation.yml")

    assert "cancel-in-progress: true" in release


def test_measurement_only_ragas_requires_an_explicit_target() -> None:
    makefile = _read("Makefile")
    release = _read(".github/workflows/release-validation.yml")

    assert "eval-measure:" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION=0" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION: '0'" not in release


def test_ragas_policy_has_no_dead_yaml_shadow_configuration() -> None:
    thresholds = _read("evaluation/thresholds.yaml")

    assert "ragas_judge:" not in thresholds
