from datetime import date

import pytest

from rag_ops_guard.domain.errors import EvidenceConflictError
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


def test_resolver_keeps_multiple_chunks_from_winning_document() -> None:
    current = metadata()
    resolved = EvidenceResolver().resolve(
        [evidence(meta=current, index=0, distance=0.1), evidence(meta=current, index=1, distance=0.2)],
        QueryContext(system="payments", environment="production"),
    )
    assert [item.chunk.chunk_index for item in resolved] == [0, 1]


def test_resolver_rejects_unresolvable_equal_authority_conflict() -> None:
    first = metadata(doc_id="policy-a", logical_id="policy", version="2.0")
    second = metadata(doc_id="policy-b", logical_id="policy", version="2.0")
    with pytest.raises(EvidenceConflictError, match="policy-a, policy-b"):
        EvidenceResolver().resolve(
            [evidence(meta=first), evidence(meta=second)],
            QueryContext(system="payments", environment="production"),
        )
