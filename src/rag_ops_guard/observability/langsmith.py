from __future__ import annotations

import os

from rag_ops_guard.config import Settings


def configure_langsmith(settings: Settings) -> None:
    """Propagate pydantic settings to LangChain/LangGraph tracing environment."""
    enabled = settings.langsmith_tracing and bool(settings.langsmith_api_key)

    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    else:
        os.environ.pop("LANGSMITH_API_KEY", None)

    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id
    else:
        os.environ.pop("LANGSMITH_WORKSPACE_ID", None)
