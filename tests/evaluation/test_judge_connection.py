from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ops_guard.config import get_settings
from rag_ops_guard.evaluation.judge import judge_connection_from_env


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text("[]", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_local_judge_uses_runtime_endpoint_without_external_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "local")
    monkeypatch.setenv("LLM_MODEL", "runtime-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.delenv("RAGAS_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("RAGAS_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)

    connection = judge_connection_from_env(_dataset(tmp_path))

    assert connection.identity.provider == "local"
    assert connection.identity.model == "runtime-model"
    assert connection.base_url == "http://localhost:8080/v1"
    assert connection.api_key == "local"


def test_external_openai_compatible_judge_requires_explicit_endpoint_and_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RAGAS_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "judge-model")
    monkeypatch.delenv("RAGAS_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)

    with pytest.raises(ValidationError, match="ragas_judge_base_url"):
        judge_connection_from_env(_dataset(tmp_path))

    monkeypatch.setenv("RAGAS_JUDGE_BASE_URL", "https://judge.example/v1")
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="ragas_judge_api_key"):
        judge_connection_from_env(_dataset(tmp_path))

    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "secret-value")
    get_settings.cache_clear()
    connection = judge_connection_from_env(_dataset(tmp_path))

    assert connection.identity.provider == "openai"
    assert connection.identity.model == "judge-model"
    assert connection.base_url == "https://judge.example/v1"
    assert connection.api_key == "secret-value"
