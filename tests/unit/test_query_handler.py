from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

from rag_ops_guard.domain.models import QueryResponse, ResponseOutcome
from rag_ops_guard.handlers import query


class _FailingAgent:
    def invoke(self, request: object) -> object:
        del request
        raise RuntimeError("embedding backend unavailable")


class _SuccessfulAgent:
    def invoke(self, request: object) -> QueryResponse:
        del request
        return QueryResponse(
            request_id="req-ok",
            outcome=ResponseOutcome.INSUFFICIENT_EVIDENCE,
            message="No admitted evidence.",
            route="uncertain",
        )


def _effective(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "config_hash": "abcdef12" + "0" * 56,
        "source": "dynamodb",
        "revision_no": 7,
        "stale": False,
        "stale_age_s": 0.0,
        "fail_closed": False,
        "db_unavailable": False,
        "revision_age_s": 12.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quiet_observability(monkeypatch) -> tuple[MagicMock, MagicMock, list[str], list[tuple[str, float]]]:
    logger = MagicMock()
    metrics = MagicMock()
    counts: list[str] = []
    measures: list[tuple[str, float]] = []
    monkeypatch.setattr(query, "QUERY_LOGGER", logger)
    monkeypatch.setattr(query, "QUERY_METRICS", metrics)
    monkeypatch.setattr(
        query,
        "add_count",
        lambda _metrics, name, value=1.0: counts.append(name),
    )
    monkeypatch.setattr(
        query,
        "add_milliseconds",
        lambda _metrics, name, value: measures.append((name, value)),
    )
    monkeypatch.setattr(
        query,
        "add_seconds",
        lambda _metrics, name, value: measures.append((name, value)),
        raising=False,
    )
    monkeypatch.setattr(query, "resolve_effective_config", lambda: _effective())
    return logger, metrics, counts, measures


def test_unexpected_query_failure_returns_diagnostic_500_in_local(monkeypatch) -> None:
    logger, _, _, _ = _quiet_observability(monkeypatch)
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


def test_phase2_query_emits_required_config_dimensions_metrics_and_trace_metadata(
    monkeypatch,
) -> None:
    _, metrics, counts, measures = _quiet_observability(monkeypatch)
    run_tree = SimpleNamespace(metadata={})
    inherited_metadata: list[dict[str, object]] = []

    @contextmanager
    def fake_tracing_context(*, metadata: dict[str, object]):
        inherited_metadata.append(dict(metadata))
        yield

    monkeypatch.setattr(query, "get_current_run_tree", lambda: run_tree, raising=False)
    monkeypatch.setattr(query, "tracing_context", fake_tracing_context, raising=False)
    monkeypatch.setattr(query, "conversation_agent", lambda: _SuccessfulAgent())

    response = query.handler(
        {"body": json.dumps({"question": "hello", "context": {}})},
        object(),
    )

    assert response["statusCode"] == 200
    metrics.add_dimension.assert_any_call(name="config_revision", value="7")
    metrics.add_dimension.assert_any_call(name="config_hash", value="abcdef12")
    metrics.add_dimension.assert_any_call(name="config_source", value="dynamodb")
    assert "ConfigCacheMiss" in counts
    assert "ConfigDbUnavailable" not in counts
    assert any(name == "ConfigResolveLatencyMs" for name, _ in measures)
    assert ("ConfigRevisionAge", 12.5) in measures
    expected_metadata = {
        "config_revision": 7,
        "config_hash": "abcdef12",
        "config_source": "dynamodb",
    }
    assert run_tree.metadata == expected_metadata
    assert inherited_metadata == [expected_metadata]


def test_phase2_query_reports_stale_cache_and_dynamodb_unavailability(monkeypatch) -> None:
    _, metrics, counts, _ = _quiet_observability(monkeypatch)
    monkeypatch.setattr(
        query,
        "resolve_effective_config",
        lambda: _effective(
            source="cache",
            stale=True,
            stale_age_s=40.0,
            db_unavailable=True,
        ),
    )
    monkeypatch.setattr(query, "conversation_agent", lambda: _SuccessfulAgent())

    response = query.handler(
        {"body": json.dumps({"question": "hello", "context": {}})},
        object(),
    )

    assert response["statusCode"] == 200
    metrics.add_dimension.assert_any_call(name="config_source", value="cache")
    assert "ConfigCacheHit" in counts
    assert "ConfigStaleServed" in counts
    assert "ConfigDbUnavailable" in counts
