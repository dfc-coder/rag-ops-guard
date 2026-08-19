from __future__ import annotations

import inspect

from pydantic import BaseModel

from rag_ops_guard.config import Settings


def test_settings_is_a_pure_pydantic_model() -> None:
    assert issubclass(Settings, BaseModel)
    assert all(base.__module__ != "pydantic_settings.main" for base in Settings.__mro__)


def test_settings_module_has_no_implicit_environment_loader() -> None:
    source = inspect.getsource(__import__("rag_ops_guard.config", fromlist=["Settings"]))
    assert "pydantic_settings" not in source
    assert "BaseSettings" not in source
    assert "SettingsConfigDict" not in source
    assert "env_file" not in source


def test_process_environment_does_not_mutate_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "must-not-be-loaded-from-process-environment")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-loaded-from-process-environment")

    settings = Settings()

    assert settings.llm_model == "qwen3.5-2b-unsloth-ud-q4-k-xl"
    assert settings.aws_secret_access_key.get_secret_value() == "test"
