from __future__ import annotations

import os

from rag_ops_guard.config import Settings
from rag_ops_guard.observability.langsmith import configure_langsmith


def test_configure_langsmith_exports_runtime_environment(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_WORKSPACE_ID", raising=False)

    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_project="rag-ops-guard-test",
        langsmith_endpoint="https://api.smith.langchain.com",
        langsmith_api_key="test-key",
        langsmith_workspace_id="workspace-test",
    )
    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "rag-ops-guard-test"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"
    assert os.environ["LANGSMITH_API_KEY"] == "test-key"
    assert os.environ["LANGSMITH_WORKSPACE_ID"] == "workspace-test"


def test_configure_langsmith_stays_disabled_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    settings = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key=None,
    )
    configure_langsmith(settings)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ
