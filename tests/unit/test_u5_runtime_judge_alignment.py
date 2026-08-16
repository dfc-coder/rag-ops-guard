from rag_ops_guard.config import Settings
from scripts.local import provision


def test_lambda_runtime_model_matches_application_model() -> None:
    """SPEC-5.3: physical Lambda and the application use one generation-model alias."""
    expected = Settings(_env_file=None).llm_model

    assert provision.LLM_MODEL == expected
    assert provision.lambda_environment()["LLM_MODEL"] == expected
