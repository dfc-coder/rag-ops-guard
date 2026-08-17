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


def test_physical_generation_artifact_is_unsloth_qwen35_08b_dynamic_quant() -> None:
    filename = "Qwen3.5-0.8B-UD-Q4_K_XL.gguf"
    expected_sha = "3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5"
    compose = _read("docker/docker-compose.yml")
    makefile = _read("Makefile")
    downloader = _read("scripts/download_models.py")

    assert f"/models/{filename}" in compose
    assert f"MODEL_FILES={filename}" in makefile
    assert "huggingface.co/unsloth/Qwen3.5-0.8B-GGUF" in downloader
    assert filename in downloader
    assert expected_sha in downloader
    assert "Qwen3.5-0.8B-Q8_0.gguf" not in compose
    assert "ggml-org/Qwen3.5-0.8B-GGUF" not in downloader


def test_unsloth_qwen35_profile_matches_non_thinking_agent_runtime() -> None:
    compose = _read("docker/docker-compose.yml")
    env = _read(".env.example")

    assert '${LLAMA_CTX_SIZE:-16384}' in compose
    assert "LLAMA_CTX_SIZE=16384" in env
    assert "--kv-unified" in compose
    assert compose.count("q8_0") >= 2
    assert "--chat-template-kwargs" in compose
    assert '{"enable_thinking":false}' in compose
    assert "--reasoning" not in compose
    assert '--temp\n      - "0.7"' in compose
    assert '--top-p\n      - "0.8"' in compose
    assert '--top-k\n      - "20"' in compose
    assert '--min-p\n      - "0.0"' in compose
    assert '--presence-penalty\n      - "1.5"' in compose
    assert '--repeat-penalty\n      - "1.0"' in compose
    assert "--jinja" in compose


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
    reranker = _read("src/rag_ops_guard/adapters/reranking/llamacpp_reranker.py")

    assert "physical-ready:" in makefile
    assert "OVMS_NETWORK=$(RAG_OPS_NETWORK)" in makefile
    assert "LAMBDA_OPENVINO_ENV" in makefile
    assert "LAMBDA_EMBEDDING_BASE_URL" in provision
    assert "LAMBDA_RERANKER_BASE_URL" in provision
    assert "EMBEDDING_TIMEOUT_SECONDS" in provision
    assert "make physical-ready" in release
    assert "OpenVINO/Qwen3-Embedding-0.6B-int8-ov" in release
    assert "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov" in release
    assert "_QWEN_SEQ_CLS_PREFIX" in reranker
    assert "<Document>:" in reranker
    assert "make local-up" not in release


def test_physical_admission_validation_uses_labelled_calibration_first() -> None:
    makefile = _read("Makefile")
    calibrator = _read("scripts/calibrate_relevance_floors.py")
    validator = _read("scripts/validate_retrieval.py")

    assert "physical-relevance-calibrate: physical-up" in makefile
    assert "--env-file .local/relevance-floors.env" in makefile
    assert "physical-ready: physical-relevance-calibrate package-lambda" in makefile
    assert makefile.index("source .local/relevance-floors.env") < makefile.index(
        "uv run python scripts/validate_retrieval.py", makefile.index("physical-ready:")
    )
    assert "expected_grounded_score" in calibrator
    assert "A high score on the wrong document must never count" in calibrator
    assert "retrieval-calibration-v2.json" in calibrator
    assert "retrieval-calibration-v2.json" in validator


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


def test_physical_validation_preserves_running_evidence() -> None:
    release = _read(".github/workflows/release-validation.yml")

    assert "cancel-in-progress: false" in release


def test_automatic_physical_workflow_measures_ragas_without_claiming_release_readiness() -> None:
    makefile = _read("Makefile")
    release = _read(".github/workflows/release-validation.yml")

    assert "eval-measure:" in makefile
    assert "physical-eval-measure:" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION=0" in makefile
    assert "RAGAS_REQUIRE_CALIBRATION: '0'" in release
    assert "External release readiness remains fail-closed" in release
    assert "physical-eval:" in makefile
    strict_block = makefile.split("physical-eval:", maxsplit=1)[1].split(
        "physical-smoke:", maxsplit=1
    )[0]
    assert "RAGAS_REQUIRE_CALIBRATION=0" not in strict_block


def test_ragas_policy_has_no_dead_yaml_shadow_configuration() -> None:
    thresholds = _read("evaluation/thresholds.yaml")

    assert "ragas_judge:" not in thresholds
