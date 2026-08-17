from rag_ops_guard.config import Settings


def test_settings_defaults_define_local_reproducible_profile() -> None:
    settings = Settings(_env_file=None)
    assert settings.app_env == "local"
    assert settings.aws_endpoint_url == "http://localhost:4566"
    assert settings.vector_dimension == 1024
    assert settings.retrieval_top_k == 20
    assert settings.retrieval_context_k == 4
    assert settings.retrieval_domain_min_relevance == 0.5
    assert settings.retrieval_min_relevance == 0.5
    assert settings.llm_model == "qwen3.5-0.8b-rag"
    assert settings.llm_analysis_max_tokens == 128
    assert settings.llm_answer_max_tokens == 512
    assert settings.llm_analysis_max_tokens < settings.llm_answer_max_tokens
    assert settings.llm_timeout_seconds == 60.0
    assert settings.llm_temperature == 0.7
    assert settings.llm_top_p == 0.8
    assert settings.llm_top_k == 20
    assert settings.llm_min_p == 0.0
    assert settings.llm_presence_penalty == 1.5
    assert settings.llm_repeat_penalty == 1.0
    assert settings.embedding_model == "qwen3-embedding-0.6b"
    assert settings.reranker_model == "qwen3-reranker-0.6b"
    assert settings.langsmith_endpoint == "https://api.smith.langchain.com"
