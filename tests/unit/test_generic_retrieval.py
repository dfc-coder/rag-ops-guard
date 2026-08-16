from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import (
    FakeEmbeddingProvider,
    FakeObjectStore,
    FakeReranker,
    FakeVectorStore,
)


def test_generic_document_without_governance_metadata_remains_searchable() -> None:
    """SPEC-2.4: optional governance cannot make a generic document unretrievable."""
    meta = metadata(
        doc_id="generic-notes",
        logical_id="generic-notes",
        status=None,
        effective_date=None,
        system=None,
        environment=None,
        document_type=None,
        authority=None,
    )
    meta.title = "Architecture Notes"
    item = evidence(meta=meta, text="The renderer uses a small intermediate document model.")
    objects = FakeObjectStore()
    objects.put_text(
        "chunks/generic-notes/2.0/chunk-000.json",
        item.chunk.model_dump_json(),
    )
    search = KnowledgeSearch(
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[item]),
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=FakeReranker(default_score=0.95, default_relevant=True),
        candidate_k=5,
        context_k=2,
    )

    result = search.search("Architecture Notes renderer", QueryContext())

    assert result.supported is True
    assert [hit.chunk.logical_id for hit in result.admitted] == ["generic-notes"]
