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


def test_local_provision_redeploys_cdk_then_runs_connectivity_gate() -> None:
    makefile = _read("Makefile")
    block = makefile.split("local-provision: package-lambda", maxsplit=1)[1].split(
        "\nseed:", maxsplit=1
    )[0]

    assert 'RAG_OPS_INFRA_TARGET=local' in block
    assert '$(CDK_LOCAL) deploy $(CDK_LOCAL_STACK)' in block
    assert 'scripts/local/provision.py' in block
    assert 'scripts/local/connectivity.py' in block
