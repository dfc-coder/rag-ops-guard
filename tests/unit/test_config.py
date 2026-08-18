import pytest
from pydantic import ValidationError

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
    assert settings.llm_model == "qwen3.5-2b-unsloth-ud-q4-k-xl"
    assert settings.llm_analysis_max_tokens == 128
    assert settings.llm_answer_max_tokens == 512
    assert settings.llm_analysis_max_tokens < settings.llm_answer_max_tokens
    assert settings.llm_timeout_seconds == 60.0
    assert settings.embedding_model == "qwen3-embedding-0.6b"
    assert settings.reranker_model == "qwen3-reranker-0.6b"
    assert settings.langsmith_endpoint == "https://api.smith.langchain.com"


def test_unknown_env_var_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retrieval_min_relevanceX=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dimension": 768},
        {"chunk_overlap": 400},
        {"retrieval_context_k": 20, "retrieval_top_k": 4},
        {"llm_analysis_max_tokens": 512, "llm_answer_max_tokens": 128},
    ],
)
def test_cross_field_invariants_are_enforced(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **kwargs)


def test_embedding_and_vector_dimension_must_match_exactly() -> None:
    settings = Settings(_env_file=None, vector_dimension=768, embedding_dimension=768)
    assert settings.embedding_dimension == settings.vector_dimension


def test_min_relevance_below_domain_floor_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            retrieval_min_relevance=0.4,
            retrieval_domain_min_relevance=0.6,
        )


def test_min_relevance_equal_or_above_domain_floor_is_accepted() -> None:
    equal = Settings(_env_file=None)
    assert equal.retrieval_min_relevance == equal.retrieval_domain_min_relevance
    above = Settings(
        _env_file=None,
        retrieval_min_relevance=0.6,
        retrieval_domain_min_relevance=0.5,
    )
    assert above.retrieval_min_relevance > above.retrieval_domain_min_relevance


def test_local_env_cannot_point_at_real_aws() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="local", aws_endpoint_url="https://s3.amazonaws.com")


def test_aws_env_cannot_point_at_loopback() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="aws", aws_endpoint_url="http://localhost:4566")


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://evil.example.com/v1",
        "file:///etc/passwd",
    ],
)
def test_model_base_urls_reject_non_local_targets_in_local_env(url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="local", llm_base_url=url)


def test_container_service_hosts_are_allowed_locally() -> None:
    settings = Settings(_env_file=None, llm_base_url="http://llama-gen:8080/v1")
    assert settings.llm_base_url.startswith("http://llama-gen")


def test_physical_lambda_openvino_host_is_allowed_locally() -> None:
    settings = Settings(
        _env_file=None,
        embedding_base_url="http://rag-ops-ovms-rag:8000/v3",
        reranker_base_url="http://rag-ops-ovms-rag:8000/v3",
    )
    assert settings.embedding_base_url == "http://rag-ops-ovms-rag:8000/v3"
    assert settings.reranker_base_url == "http://rag-ops-ovms-rag:8000/v3"


SENTINEL_LANGSMITH = "lsv2-sentinela-no-debe-aparecer"
SENTINEL_AWS = "AKIA-sentinela-no-debe-aparecer"
SENTINEL_JUDGE = "sk-sentinela-no-debe-aparecer"


def test_no_plaintext_secret_in_any_serialization() -> None:
    settings = Settings(
        _env_file=None,
        langsmith_api_key=SENTINEL_LANGSMITH,
        aws_secret_access_key=SENTINEL_AWS,
        ragas_judge_provider="openai",
        ragas_judge_base_url="https://api.openai.com/v1",
        ragas_judge_api_key=SENTINEL_JUDGE,
    )
    surfaces = [
        repr(settings),
        str(settings),
        str(settings.model_dump()),
        settings.model_dump_json(),
    ]
    for surface in surfaces:
        for sentinel in (SENTINEL_LANGSMITH, SENTINEL_AWS, SENTINEL_JUDGE):
            assert sentinel not in surface


def test_secret_is_still_retrievable_explicitly() -> None:
    settings = Settings(_env_file=None, langsmith_api_key=SENTINEL_LANGSMITH)
    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == SENTINEL_LANGSMITH


def test_remote_judge_requires_base_url_and_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ragas_judge_provider="openai")
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            ragas_judge_provider="openai",
            ragas_judge_base_url="https://api.openai.com/v1",
        )


def test_local_judge_falls_back_to_runtime_model() -> None:
    settings = Settings(_env_file=None)
    assert settings.resolved_ragas_judge_model == settings.llm_model
    assert settings.resolved_ragas_judge_base_url == settings.llm_base_url


def test_explicit_judge_model_overrides_runtime_model() -> None:
    settings = Settings(_env_file=None, ragas_judge_model="otro-modelo")
    assert settings.resolved_ragas_judge_model == "otro-modelo"


def test_repo_env_example_loads_under_forbid() -> None:
    settings = Settings(_env_file=".env.example")
    assert settings.app_env in ("local", "local-observed", "ci", "aws")
