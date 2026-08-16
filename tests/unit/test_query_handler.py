from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_ops_guard.handlers import query


class _FailingAgent:
    def invoke(self, request: object) -> object:
        del request
        raise RuntimeError("embedding backend unavailable")


def _quiet_observability(monkeypatch) -> MagicMock:
    logger = MagicMock()
    monkeypatch.setattr(query, "QUERY_LOGGER", logger)
    monkeypatch.setattr(query, "add_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(query, "add_milliseconds", lambda *args, **kwargs: None)
    monkeypatch.setattr(query, "QUERY_METRICS", SimpleNamespace(flush_metrics=lambda: None))
    return logger


def test_unexpected_query_failure_returns_diagnostic_500_in_local(monkeypatch) -> None:
    logger = _quiet_observability(monkeypatch)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setattr(query, "conversation_agent", lambda: _FailingAgent())

    response = query.handler(
        {"body": json.dumps({"question": "hello", "context": {}})},
        object(),
    )

    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {
        "error": "internal_error",
        "detail": "embedding backend unavailable",
    }
    logger.exception.assert_called_once()


def test_unexpected_query_failure_does_not_expose_detail_outside_local(monkeypatch) -> None:
    _quiet_observability(monkeypatch)
    monkeypatch.setenv("APP_ENV", "aws")
    monkeypatch.setattr(query, "conversation_agent", lambda: _FailingAgent())

    response = query.handler(
        {"body": json.dumps({"question": "hello", "context": {}})},
        object(),
    )

    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {"error": "internal_error"}
