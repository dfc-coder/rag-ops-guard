from __future__ import annotations

from rag_ops_guard.domain.models import Evidence, QueryContext
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import (
    FakeEmbeddingProvider,
    FakeObjectStore,
    FakeReranker,
    FakeVectorStore,
)


def _item(logical_id: str, title: str, text: str, *, distance: float) -> Evidence:
    meta = metadata(doc_id=f"{logical_id}-v1", logical_id=logical_id)
    meta.title = title
    item = evidence(meta=meta, text=text, distance=distance)
    item.chunk.title = title
    return item


def _store(objects: FakeObjectStore, items: list[Evidence]) -> None:
    for item in items:
        objects.put_text(
            f"chunks/{item.chunk.logical_id}/{item.chunk.version}/chunk-000.json",
            item.chunk.model_dump_json(),
        )


def test_reranker_sees_dense_only_candidate_pushed_below_rrf_cutoff() -> None:
    common = "reintentos permitidos para una operación transitoria"
    first = _item("decoy-1", "Decoy One", common, distance=0.1)
    second = _item("decoy-2", "Decoy Two", common, distance=0.2)
    third = _item("decoy-3", "Decoy Three", common, distance=0.3)
    lexical_only = _item(
        "aaa-lexical-only",
        "Lexical Only",
        common,
        distance=0.9,
    )
    target = _item(
        "zzz-authoritative-target",
        "Authoritative Retry Policy",
        "Automatic retry is allowed a maximum of three times.",
        distance=0.4,
    )

    objects = FakeObjectStore()
    _store(objects, [first, second, third, lexical_only, target])
    reranker = FakeReranker(
        default_score=0.1,
        default_relevant=False,
        scores_by_document={"Authoritative Retry Policy": 0.99},
        relevant_by_document={"Authoritative Retry Policy": True},
    )
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[first, second, third, target]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=reranker,
        candidate_k=4,
        context_k=1,
    )

    result = search.search("reintentos permitidos", QueryContext(), query_mode="probe")

    assert result.supported is True
    assert result.admitted[0].chunk.logical_id == "zzz-authoritative-target"
    assert any(
        "Authoritative Retry Policy" in document
        for _query, documents in reranker.calls
        for document in documents
    )
