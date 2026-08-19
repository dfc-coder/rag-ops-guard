from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_query_lambda_exposes_local_only_runtime_connectivity_probe() -> None:
    handler = _read("src/rag_ops_guard/handlers/query.py")
    diagnostics = _read("src/rag_ops_guard/diagnostics/connectivity.py")

    assert '__rag_ops_probe__' in handler
    assert '"connectivity"' in handler
    assert 'APP_ENV' in handler
    assert 'probe_runtime_connectivity' in handler
    assert 'function_name' in handler

    assert 'dynamodb' in diagnostics
    assert 's3' in diagnostics
    assert 's3vectors' in diagnostics
    assert 'llm' in diagnostics
    assert 'embeddings' in diagnostics
    assert 'reranker' in diagnostics
    assert 'embed_query' in diagnostics
    assert 'grade(' in diagnostics
    assert 'query(' in diagnostics


def test_golden_preflight_requires_direct_lambda_and_api_connectivity() -> None:
    gate = _read("scripts/local/connectivity.py")
    makefile = _read("Makefile")

    assert 'connectivity:' in makefile
    assert 'scripts/local/connectivity.py' in makefile
    assert '_invoke_lambda_connectivity' in gate
    assert '_invoke_api_connectivity' in gate
    assert '_api_route_integration_problem' in gate
    assert 'Lambda topology mismatch' in gate
    assert 'runtime connectivity probe failed' in gate
    assert 'API connectivity probe failed' in gate

    golden = makefile.split("golden:\n", maxsplit=1)[1].split("\ngolden-all:", maxsplit=1)[0]
    golden_all = makefile.split("golden-all:\n", maxsplit=1)[1].split("\nlogs:", maxsplit=1)[0]
    assert '$(MAKE) connectivity' in golden
    assert '$(MAKE) connectivity' in golden_all


def test_local_provision_is_tenant_independent_and_ready_runs_authenticated_connectivity() -> None:
    makefile = _read("Makefile")
    infra = makefile.split("local-infra: package-lambda", maxsplit=1)[1].split(
        "\nlocal-provision:", maxsplit=1
    )[0]
    provision = makefile.split("local-provision: local-infra", maxsplit=1)[1].split(
        "\nseed:", maxsplit=1
    )[0]
    ready = makefile.split("physical-ready: physical-relevance-calibrate", maxsplit=1)[1].split(
        "\nphysical-eval-measure:", maxsplit=1
    )[0]

    assert 'RAG_OPS_INFRA_TARGET=local' in infra
    assert '$(CDK_LOCAL) deploy $(CDK_LOCAL_STACK)' in infra
    assert 'scripts/local/provision.py' in infra
    assert 'if [ "$(SKIP_CDK_BOOTSTRAP)" != "1" ]' in infra
    assert '$(MAKE) config-bootstrap' in provision
    assert 'scripts/local/connectivity.py' not in provision
    assert '$(MAKE) local-infra SKIP_CDK_BOOTSTRAP=1' in provision
    assert provision.index('$(MAKE) config-bootstrap') < provision.index(
        '$(MAKE) local-infra SKIP_CDK_BOOTSTRAP=1'
    )
    assert '$(MAKE) tenant-sync TENANT=$(TENANT)' in ready
    assert 'scripts/local/connectivity.py --tenant-id "$(TENANT)"' in ready


def test_local_cdk_uses_ovms_model_identity_not_generic_env_fallback() -> None:
    makefile = _read("Makefile")
    cdk_app = _read("infra/cdk/bin/rag-ops-guard.ts")

    assert 'OVMS_EMBEDDING_MODEL ?= OpenVINO/Qwen3-Embedding-0.6B-int8-ov' in makefile
    assert 'OVMS_RERANKER_MODEL ?= OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov' in makefile
    assert 'OVMS_EMBEDDING_MODEL' in makefile.split("export ", maxsplit=1)[1].split("\n", maxsplit=1)[0]
    assert 'OVMS_RERANKER_MODEL' in makefile.split("export ", maxsplit=1)[1].split("\n", maxsplit=1)[0]

    assert (
        'embeddingModel: local ? process.env.OVMS_EMBEDDING_MODEL : process.env.EMBEDDING_MODEL'
        in cdk_app
    )
    assert (
        'rerankerModel: local ? process.env.OVMS_RERANKER_MODEL : process.env.RERANKER_MODEL'
        in cdk_app
    )
    assert 'process.env.LAMBDA_EMBEDDING_MODEL ?? process.env.EMBEDDING_MODEL' not in cdk_app
    assert 'process.env.LAMBDA_RERANKER_MODEL ?? process.env.RERANKER_MODEL' not in cdk_app
