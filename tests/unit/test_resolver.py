from datetime import date

from rag_ops_guard.domain.models import DocumentStatus, QueryContext
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata


def test_resolver_excludes_deprecated_document() -> None:
    old = metadata(
        doc_id="payment-retry-policy-v1",
        version="1.0",
        status=DocumentStatus.DEPRECATED,
        effective_date=date(2025, 1, 1),
    )
    current = metadata(supersedes=["payment-retry-policy-v1"])
    resolved = EvidenceResolver().resolve(
        [evidence(meta=old, distance=0.01), evidence(meta=current, distance=0.2)],
        QueryContext(system="payments", environment="production"),
    )
    assert {item.chunk.metadata.id for item in resolved} == {"payment-retry-policy-v2"}


def test_resolver_filters_requested_system() -> None:
    payments = evidence(meta=metadata(system="payments"))
    calypso = evidence(
        meta=metadata(doc_id="calypso-runbook", logical_id="calypso-runbook", system="calypso")
    )
    resolved = EvidenceResolver().resolve([payments, calypso], QueryContext(system="calypso"))
    assert [item.chunk.metadata.system for item in resolved] == ["calypso"]
