from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase2_data_plane_uses_effective_config() -> None:
    app = (ROOT / "src/rag_ops_guard/app.py").read_text(encoding="utf-8")
    query = (ROOT / "src/rag_ops_guard/handlers/query.py").read_text(encoding="utf-8")

    assert "resolve_effective_config" in app
    assert "resolve_effective_config" in query
    assert "config_hash" in query


def test_phase2_calibration_publishes_and_shadow_check_is_blocking() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    calibrator = (ROOT / "scripts/calibrate_relevance_floors.py").read_text(encoding="utf-8")

    assert "--publish" in makefile
    assert "publish_config_revision" in calibrator
    shadow_step = ci.split("Config shadow check", 1)[1].split("- if:", 1)[0]
    assert "continue-on-error: true" not in shadow_step


def test_phase2_lambda_bootstrap_does_not_duplicate_runtime_tuning() -> None:
    stack = (ROOT / "infra/cdk/lib/rag-ops-guard-stack.ts").read_text(encoding="utf-8")

    start = stack.index("const commonEnvironment =")
    end = stack.index("const ingest =")
    body = stack[start:end]
    for key in (
        "RETRIEVAL_TOP_K",
        "RETRIEVAL_CONTEXT_K",
        "RETRIEVAL_DOMAIN_MIN_RELEVANCE",
        "RETRIEVAL_MIN_RELEVANCE",
        "CHUNK_TOKENS",
        "CHUNK_OVERLAP",
        "LLM_TEMPERATURE",
    ):
        assert key not in body

    assert "CONFIG_SOURCE: 'db'" in body
    assert "CONFIG_TABLE:" in body
    assert "CONFIG_HASH:" in body
