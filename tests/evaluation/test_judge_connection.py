from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_ops_guard.config import Settings
from rag_ops_guard.evaluation.judge import judge_connection_from_settings


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "golden.json"
    path.write_text("[]", encoding="utf-8")
    return path


def test_local_judge_uses_explicit_runtime_settings(tmp_path: Path) -> None:
    settings = Settings(
        ragas_judge_provider="local",
        llm_model="runtime-model",
        llm_base_url="http://localhost:8080/v1",
    )

    connection = judge_connection_from_settings(settings, _dataset(tmp_path))

    assert connection.identity.provider == "local"
    assert connection.identity.model == "runtime-model"
    assert connection.base_url == "http://localhost:8080/v1"
    assert connection.api_key == "local"


def test_external_openai_judge_requires_explicit_endpoint_and_secret(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="ragas_judge_base_url"):
        Settings(ragas_judge_provider="openai", ragas_judge_model="judge-model")

    with pytest.raises(ValidationError, match="ragas_judge_api_key"):
        Settings(
            ragas_judge_provider="openai",
            ragas_judge_model="judge-model",
            ragas_judge_base_url="https://judge.example/v1",
        )

    settings = Settings(
        ragas_judge_provider="openai",
        ragas_judge_model="judge-model",
        ragas_judge_base_url="https://judge.example/v1",
        ragas_judge_api_key="secret-value",
    )
    connection = judge_connection_from_settings(settings, _dataset(tmp_path))

    assert connection.identity.provider == "openai"
    assert connection.identity.model == "judge-model"
    assert connection.base_url == "https://judge.example/v1"
    assert connection.api_key == "secret-value"
