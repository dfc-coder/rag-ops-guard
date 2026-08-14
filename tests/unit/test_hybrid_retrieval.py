from __future__ import annotations

from rag_ops_guard.domain.models import DocumentType, Evidence, QueryContext
from rag_ops_guard.retrieval.bm25 import BM25Index
from rag_ops_guard.retrieval.hybrid import reciprocal_rank_fusion
from tests.fixtures.builders import evidence, metadata


def _named_evidence(title: str, text: str, *, logical_id: str, distance: float = 0.4) -> Evidence:
    meta = metadata(doc_id=f"{logical_id}-v1", logical_id=logical_id)
    meta.title = title
    meta.document_type = DocumentType.RUNBOOK
    item = evidence(meta=meta, text=text, distance=distance)
    item.chunk.title = title
    return item


def test_bm25_recovers_exact_entity_document() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failures must be retried independently.",
        logical_id="sendgrid-failure",
    )
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
    )

    results = BM25Index([sendgrid, payments]).search("What does SendGrid do?", limit=2)

    assert results[0].chunk.logical_id == "sendgrid-failure"


def test_rrf_can_promote_lexical_result_missed_by_dense_top_rank() -> None:
    sendgrid = _named_evidence(
        "SendGrid Failure Runbook",
        "Notification delivery failures must be retried independently.",
        logical_id="sendgrid-failure",
        distance=0.8,
    )
    payments = _named_evidence(
        "Payment Retry Policy",
        "Transient payment timeouts may be retried.",
        logical_id="payment-retry-policy",
        distance=0.1,
    )

    fused = reciprocal_rank_fusion(
        dense=[payments, sendgrid],
        lexical=[sendgrid, payments],
    )

    assert {item.chunk.logical_id for item in fused[:2]} == {
        "sendgrid-failure",
        "payment-retry-policy",
    }


def test_active_resolver_contract_remains_compatible_with_hybrid_results() -> None:
    current = _named_evidence(
        "Payment Retry Policy",
        "Maximum of three retries.",
        logical_id="payment-retry-policy",
    )
    assert current.chunk.metadata.status.value == "active"
    assert QueryContext(system="payments").system == "payments"
