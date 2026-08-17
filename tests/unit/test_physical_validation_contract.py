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


def test_physical_floci_uses_standard_lambda_and_api_contract() -> None:
    compose = _read("docker/docker-compose.yml")
    hosted_ci = _read(".github/workflows/ci.yml")
    provision = _read("scripts/local/provision.py")

    assert "floci/floci:1.6.0" in compose
    assert "floci/floci:1.6.0" in hosted_ci
    assert "floci:override-id" not in provision
    assert '"hot-reload"' not in provision
    assert "S3_LAMBDA_CODE_BUCKET" in provision
    assert 'Code={"S3Bucket": LAMBDA_CODE_BUCKET, "S3Key": code_key}' in provision
    assert "/execute-api/{api_id}/" in provision


def test_physical_floci_matches_documented_rootless_podman_contract() -> None:
    compose = _read("docker/docker-compose.yml")

    assert 'FLOCI_DOCKER_DOCKER_HOST: unix:///var/run/docker.sock' in compose
    assert 'FLOCI_SERVICES_DOCKER_NETWORK: ${RAG_OPS_NETWORK:-rag-ops-net}' in compose
    assert 'FLOCI_SERVICES_LAMBDA_DOCKER_NETWORK: ${RAG_OPS_NETWORK:-rag-ops-net}' in compose
    assert 'FLOCI_SERVICES_LAMBDA_DOCKER_HOST_OVERRIDE: floci' in compose
    assert '"${PODMAN_SOCKET}:/var/run/docker.sock:z"' in compose
    assert "security_opt:" in compose
    assert "- label=disable" in compose


def test_physical_profile_uses_openvino_for_embedding_and_reranking() -> None:
    makefile = _read("Makefile")
    provision = _read("scripts/local/provision.py")
    release = _read(".github/workflows/release-validation.yml")

    assert "physical-ready:" in makefile
    assert "OVMS_NETWORK=$(RAG_OPS_NETWORK)" in makefile
    assert "LAMBDA_OPENVINO_ENV" in makefile
    assert "LAMBDA_EMBEDDING_BASE_URL" in provision
    assert "LAMBDA_RERANKER_BASE_URL" in provision
    assert "EMBEDDING_TIMEOUT_SECONDS" in provision
    assert "make physical-ready" in release
    assert "OpenVINO/Qwen3-Embedding-0.6B-int8-ov" in release
    assert "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov" in release
    assert "make local-up" not in release


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


def test_release_preserves_running_physical_evidence() -> None:
    release = _read(".github/workflows/release-validation.yml")

    assert "cancel-in-progress: false" in release


def test_measurement_only_ragas_requires_an_explicit_target() -> None:
    makefile = _read("Makefile")
    release = _read(".github/workflows/release-validation.yml")

    assert "eval-measure:" in makefile
    assert "physical-eval-measure:" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION=0" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION: '0'" not in release


def test_ragas_policy_has_no_dead_yaml_shadow_configuration() -> None:
    thresholds = _read("evaluation/thresholds.yaml")

    assert "ragas_judge:" not in thresholds
