from rag_ops_guard.retrieval.reranker import rerank_evidence
from tests.fixtures.builders import evidence, metadata


def test_reranker_promotes_candidate_with_specific_query_terms() -> None:
    generic_meta = metadata(doc_id="generic", logical_id="generic").model_copy(
        update={"title": "Payment Processing Flow"}
    )
    retry_meta = metadata(doc_id="retry", logical_id="retry").model_copy(
        update={"title": "Payment Retry Policy"}
    )

    generic = evidence(
        meta=generic_meta,
        distance=0.05,
        text="The Payment API routes transactions to several downstream systems.",
    )
    retry = evidence(
        meta=retry_meta,
        distance=0.12,
        text="Transient Calypso timeouts may be retried automatically three times.",
    )

    ranked = rerank_evidence("cuantos reintentos tiene Calypso en la Payment API?", [generic, retry])

    assert ranked[0].chunk.metadata.id == "retry"
