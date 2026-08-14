from __future__ import annotations

from rag_ops_guard.agent.catalog import KnowledgeCatalog
from rag_ops_guard.domain.models import DocumentStatus, DocumentType, QueryContext
from tests.fixtures.builders import evidence, metadata
from tests.fixtures.fakes import FakeObjectStore


def _store_chunk(objects: FakeObjectStore, item) -> None:
    key = (
        f"chunks/{item.chunk.logical_id}/{item.chunk.version}/"
        f"chunk-{item.chunk.chunk_index:03d}.json"
    )
    objects.put_text(key, item.chunk.model_dump_json())


def test_catalog_lists_active_documents_once_and_skips_deprecated() -> None:
    objects = FakeObjectStore()
    active_meta = metadata(logical_id="calypso-api", doc_id="calypso-api-v3", system="calypso")
    active_meta.title = "Calypso Integration API"
    active_meta.document_type = DocumentType.API
    active = evidence(meta=active_meta)
    second_chunk = evidence(meta=active_meta, index=1)

    old_meta = metadata(
        logical_id="old-policy",
        doc_id="old-policy-v1",
        status=DocumentStatus.DEPRECATED,
    )
    old = evidence(meta=old_meta)
    for item in (active, second_chunk, old):
        _store_chunk(objects, item)

    entries = KnowledgeCatalog(objects).entries()

    assert [(item.title, item.document_type) for item in entries] == [
        ("Calypso Integration API", "api")
    ]


def test_catalog_respects_context_and_renders_real_titles() -> None:
    objects = FakeObjectStore()
    calypso_meta = metadata(logical_id="calypso-api", doc_id="calypso-v3", system="calypso")
    calypso_meta.title = "Calypso Integration API"
    payments_meta = metadata(logical_id="payments-api", doc_id="payments-v2", system="payments")
    payments_meta.title = "Payments API v2"
    for item in (evidence(meta=calypso_meta), evidence(meta=payments_meta)):
        _store_chunk(objects, item)

    rendered = KnowledgeCatalog(objects).render(
        "¿Qué documentación tienes disponible?",
        QueryContext(system="calypso"),
    )

    assert "Calypso Integration API" in rendered
    assert "Payments API v2" not in rendered
