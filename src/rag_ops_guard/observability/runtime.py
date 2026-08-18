from __future__ import annotations

from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.metrics import MetricUnit

QUERY_LOGGER = Logger(service="rag-ops-guard-query")
INGEST_LOGGER = Logger(service="rag-ops-guard-ingest")
QUERY_METRICS = Metrics(namespace="RagOpsGuard", service="query")
INGEST_METRICS = Metrics(namespace="RagOpsGuard", service="ingest")


def add_count(metrics: Metrics, name: str, value: float = 1.0) -> None:
    metrics.add_metric(name=name, unit=MetricUnit.Count, value=value)


def add_milliseconds(metrics: Metrics, name: str, value: float) -> None:
    metrics.add_metric(name=name, unit=MetricUnit.Milliseconds, value=value)


def add_seconds(metrics: Metrics, name: str, value: float) -> None:
    metrics.add_metric(name=name, unit=MetricUnit.Seconds, value=value)


def config_observability_fields(
    *,
    revision_no: int | None,
    config_hash: str,
    source: str,
) -> dict[str, str | int | None]:
    return {
        "config_revision": revision_no,
        "config_hash": config_hash[:8],
        "config_source": source,
    }


def set_config_dimensions(
    metrics: Metrics,
    *,
    revision_no: int | None,
    config_hash: str,
    source: str,
) -> None:
    fields = config_observability_fields(
        revision_no=revision_no,
        config_hash=config_hash,
        source=source,
    )
    for name, value in fields.items():
        metrics.add_dimension(name=name, value="none" if value is None else str(value))
