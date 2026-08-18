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
    ops = _read("scripts/local/ops.py")
    makefile = _read("Makefile")

    assert 'connectivity' in makefile
    assert 'ops.py connectivity' in makefile
    assert '_connectivity_problems' in ops
    assert '_invoke_lambda_connectivity' in ops
    assert '_invoke_api_connectivity' in ops
    assert '_api_route_integration_problem' in ops
    assert 'Lambda topology mismatch' in ops
    assert 'runtime connectivity probe failed' in ops
    assert 'API connectivity probe failed' in ops


def test_local_provision_runs_connectivity_gate_after_redeploy() -> None:
    makefile = _read("Makefile")
    block = makefile.split("local-provision: package-lambda", maxsplit=1)[1].split(
        "\nseed:", maxsplit=1
    )[0]

    assert '$(LAMBDA_OPENVINO_ENV)' in block
    assert 'ops.py connectivity' in block
