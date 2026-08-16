from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_ops_guard.handlers import ingest


class _FailingIngestionService:
    def ingest(self, s3_key: str) -> object:
        del s3_key
        raise RuntimeError("vector backend unavailable")


def _quiet_observability(monkeypatch) -> MagicMock:
    logger = MagicMock()
    monkeypatch.setattr(ingest, "INGEST_LOGGER", logger)
    monkeypatch.setattr(ingest, "add_count", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "add_milliseconds", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingest, "INGEST_METRICS", SimpleNamespace(flush_metrics=lambda: None))
    return logger


def test_unexpected_ingest_failure_returns_diagnostic_500_in_local(monkeypatch) -> None:
    logger = _quiet_observability(monkeypatch)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setattr(ingest, "ingestion_service", lambda: _FailingIngestionService())

    response = ingest.handler(
        {"body": json.dumps({"s3_key": "raw/test.md"})},
        object(),
    )

    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {
        "error": "internal_error",
        "detail": "vector backend unavailable",
    }
    logger.exception.assert_called_once()
