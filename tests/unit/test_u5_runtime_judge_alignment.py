from pathlib import Path

from rag_ops_guard.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_lambda_runtime_model_matches_application_model() -> None:
    """SPEC-5.3: physical Lambda and the application use one generation-model alias."""
    expected = Settings(_env_file=None).llm_model
    stack = (ROOT / "infra/cdk/lib/rag-ops-guard-stack.ts").read_text(encoding="utf-8")
    entrypoint = (ROOT / "infra/cdk/bin/rag-ops-guard.ts").read_text(encoding="utf-8")

    assert f"props.llmModel ?? '{expected}'" in stack
    assert "llmModel: process.env.LLM_MODEL" in entrypoint
