from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.registry import public_settings_values, registry_entries


def test_registry_mirrors_all_settings_fields() -> None:
    entries = registry_entries()
    assert set(entries) == set(Settings.model_fields)
    assert len(entries) == 56


def test_registry_mirrors_representative_bounds() -> None:
    entries = registry_entries()
    top_k = entries["retrieval_top_k"].json_schema
    ctx = entries["llama_ctx_size"].json_schema
    assert top_k["minimum"] == 1
    assert top_k["maximum"] == 100
    assert ctx["minimum"] == 512
    assert ctx["maximum"] == 131072


def test_registry_marks_exact_system_secrets() -> None:
    entries = registry_entries()
    secrets = {name for name, entry in entries.items() if entry.sensitivity == "secret"}
    assert secrets == {
        "aws_secret_access_key",
        "langsmith_api_key",
        "ragas_judge_api_key",
    }


def test_relevance_floors_are_fail_closed() -> None:
    entries = registry_entries()
    assert entries["retrieval_domain_min_relevance"].fail_mode == "fail_closed"
    assert entries["retrieval_min_relevance"].fail_mode == "fail_closed"


def test_secret_values_are_never_publication_values() -> None:
    settings = Settings(_env_file=None)
    values = public_settings_values(settings)
    assert len(values) == 53
    assert "aws_secret_access_key" not in values
    assert "langsmith_api_key" not in values
    assert "ragas_judge_api_key" not in values
