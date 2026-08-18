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


def test_make_workflows_pin_canonical_python() -> None:
    makefile = _read("Makefile")
    assert "UV_PYTHON ?= 3.12" in makefile
    assert "export UV_PYTHON " in makefile
    assert _read(".python-version").strip() == "3.12"


def test_physical_runtime_uses_one_generation_model_alias() -> None:
    alias = _env_value("LLM_MODEL")
    compose = _read("docker/docker-compose.yml")
    provision = _read("scripts/local/provision.py")
    release = _read(".github/workflows/release-validation.yml")

    assert f"- {alias}" in compose
    assert '"LLM_MODEL": LLM_MODEL' in provision
    assert f"LLM_MODEL: {alias}" in release
    assert f"RAGAS_JUDGE_MODEL: {alias}" in release


def test_physical_generation_artifact_is_unsloth_qwen35_2b_dynamic_quant() -> None:
    filename = "Qwen3.5-2B-UD-Q4_K_XL.gguf"
    expected_sha = "0af96165ea615bea39a04118d63f0b6d35908aea850ee4a51aa6151d851b8b35"
    compose = _read("docker/docker-compose.yml")
    makefile = _read("Makefile")
    downloader = _read("scripts/download_models.py")

    assert f"/models/{filename}" in compose
    assert f"MODEL_FILES={filename}" in makefile
    assert "huggingface.co/unsloth/Qwen3.5-2B-GGUF" in downloader
    assert filename in downloader
    assert expected_sha in downloader
    assert "Qwen3.5-0.8B-UD-Q4_K_XL.gguf" not in compose


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


def test_generation_contract_uses_tool_calling_plus_schema_constrained_final_output() -> None:
    makefile = _read("Makefile")
    contract = _read("scripts/validate_llama_contract.py")
    adapter = _read("src/rag_ops_guard/adapters/llm/openai_tool_calling.py")

    assert "physical-generation-contract: physical-up" in makefile
    assert "physical-relevance-calibrate: physical-generation-contract" in makefile
    assert '"tool_choice": "auto"' in contract
    assert '"response_format"' in contract
    assert '"type": "json_object"' in contract
    assert "llama schema-constrained structured response: ready" in contract
    assert "LLAMA FUNCTION-CALLING CONTRACT READY" in contract
    assert "submit_structured_response" not in contract
    assert "with_structured_output" not in adapter
    assert "response_format" in adapter
    assert "model_validate_json" in adapter
    assert "FINAL_RESPONSE_TOOL" not in adapter


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


def test_physical_floci_matches_rootless_podman_proxy_workaround_contract() -> None:
    compose = _read("docker/docker-compose.yml")
    makefile = _read("Makefile")
    bridge = _read("scripts/local/podman_api_bridge.py")

    # Floci's GraalVM native image can fail on docker-java UnixDomainSockets.
    # Keep Floci on TCP loopback inside its own network namespace; socat shares
    # that namespace and is the only container that sees a project-scoped,
    # long-lived rootless Podman Unix socket. No Docker API TCP port is exposed
    # on the host or the application network.
    assert "docker-proxy:" in compose
    assert "docker.io/alpine/socat:1.8.1.3" in compose
    assert "TCP-LISTEN:2375,fork,reuseaddr" in compose
    assert "UNIX-CONNECT:/var/run/docker.sock" in compose
    assert 'FLOCI_DOCKER_DOCKER_HOST: tcp://127.0.0.1:2375' in compose
    assert 'network_mode: "service:floci"' in compose
    assert '"${PODMAN_API_SOCKET}:/var/run/docker.sock"' in compose
    assert 'FLOCI_SERVICES_DOCKER_NETWORK: ${RAG_OPS_NETWORK:-rag-ops-net}' in compose
    assert 'FLOCI_SERVICES_LAMBDA_DOCKER_NETWORK: ${RAG_OPS_NETWORK:-rag-ops-net}' in compose
    assert 'FLOCI_SERVICES_LAMBDA_DOCKER_HOST_OVERRIDE: floci' in compose
    assert "docker-control:" not in compose
    assert "2375:2375" not in compose
    assert "security_opt:" in compose
    assert "- label=disable" in compose

    assert '"podman", "system", "service", "--time=0"' in bridge
    assert "socket.AF_UNIX" in bridge
    assert "GET /_ping HTTP/1.1" in bridge
    assert "PODMAN_API_SOCKET ?=" in makefile
    assert "podman_api_bridge.py start" in makefile
    assert "podman_api_bridge.py stop" in makefile


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
    lambda_env = provision.split("def lambda_environment", maxsplit=1)[1].split(
        "def publish_lambda_code", maxsplit=1
    )[0]
    assert "EMBEDDING_TIMEOUT_SECONDS" not in lambda_env
    assert "RERANKER_TIMEOUT_SECONDS" not in lambda_env
    assert "make physical-ready" in release
    assert "OpenVINO/Qwen3-Embedding-0.6B-int8-ov" in release
    assert "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov" in release
    assert "_QWEN_SEQ_CLS_PREFIX" in reranker
    assert "<Document>:" in reranker
    assert "make local-up" not in release


def test_physical_admission_validation_uses_one_labelled_calibration_dataset() -> None:
    makefile = _read("Makefile")
    calibrator = _read("scripts/calibrate_relevance_floors.py")
    validator = _read("scripts/validate_retrieval.py")
    dataset_name = "retrieval-relevance-calibration.json"

    assert "physical-relevance-calibrate: physical-generation-contract" in makefile
    assert "--env-file .local/relevance-floors.env" in makefile
    assert "--publish" in makefile
    assert "physical-ready: physical-relevance-calibrate package-lambda" in makefile
    physical_ready = makefile.split("physical-ready:", maxsplit=1)[1].split(
        "physical-eval-measure:", maxsplit=1
    )[0]
    assert "CONFIG_SOURCE=db" in physical_ready
    assert "source .local/relevance-floors.env" not in physical_ready
    assert "publish_config_revision" in calibrator
    assert "expected_grounded_score" in calibrator
    assert dataset_name in calibrator
    assert dataset_name in validator
    assert not (ROOT / "evaluation/datasets/retrieval-calibration-v1.json").exists()
    assert not (ROOT / "evaluation/datasets/retrieval-calibration-v2.json").exists()


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


def test_physical_validation_prioritizes_latest_candidate() -> None:
    release = _read(".github/workflows/release-validation.yml")
    assert "cancel-in-progress: true" in release


def test_physical_provisioning_propagates_langsmith_without_hardcoded_disable() -> None:
    provision = _read("scripts/local/provision.py")
    handler = _read("src/rag_ops_guard/handlers/query.py")
    release = _read(".github/workflows/release-validation.yml")
    loader = _read("scripts/local/configure_langsmith_ci.py")

    assert '_langsmith_environment()' in provision
    assert '"LANGSMITH_TRACING": "false"' not in provision
    assert '"LANGSMITH_API_KEY"' in provision
    assert '@traceable(name="rag_query", run_type="chain")' in handler
    assert "configure_langsmith_ci.py" in release
    assert "Documents/projects/rag-ops-guard/.env" in loader
    assert "::add-mask::" in loader


def test_automatic_physical_workflow_requires_34_golden_traces() -> None:
    release = _read(".github/workflows/release-validation.yml")
    verifier = _read("scripts/verify_langsmith_golden.py")
    thresholds = _read("evaluation/thresholds.yaml")

    assert "Run Golden 34 with LangSmith tracing" in release
    assert "Verify all 34 Golden traces reached LangSmith" in release
    assert "expected exactly 34 unique Golden questions" in verifier
    assert "LANGSMITH GOLDEN READY: 34/34" in verifier
    assert "case_accuracy: 1.00" in thresholds


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