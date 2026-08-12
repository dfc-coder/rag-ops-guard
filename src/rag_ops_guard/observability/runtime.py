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
