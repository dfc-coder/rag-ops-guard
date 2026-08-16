from __future__ import annotations

import pytest

from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ingestion.metadata import parse_document
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata


def test_plain_markdown_without_frontmatter_ingests() -> None:
    """SPEC-2.1 / SPEC-2.2"""
    meta, body = parse_document(
        "# Notas de arquitectura\n\nContenido.",
        filename="notas.md",
    )

    assert meta.title == "Notas de arquitectura"
    assert meta.id.startswith("doc-")
    assert meta.logical_id == meta.id
    assert meta.version == "1"
    assert meta.authority is None
    assert meta.document_type is None
    assert body == "# Notas de arquitectura\n\nContenido."


def test_plain_text_uses_filename_for_title() -> None:
    """SPEC-2.1 / SPEC-2.2"""
    meta, body = parse_document("Contenido sin headings.", filename="manual-operativo.txt")

    assert meta.title == "manual operativo"
    assert meta.authority is None
    assert body == "Contenido sin headings."


def test_unsupported_generic_extension_is_rejected() -> None:
    """SPEC-2.2"""
    with pytest.raises(DocumentValidationError, match="supported formats"):
        parse_document("contenido", filename="manual.pdf")


def test_valid_rich_frontmatter_is_preserved() -> None:
    """SPEC-2.3"""
    content = """---
id: payment-retry-policy-v2
logical_id: payment-retry-policy
title: Payment Retry Policy
version: "2.0"
status: active
effective_date: 2026-06-01
system: payments
environment: production
document_type: runbook
authority: 100
supersedes:
  - payment-retry-policy-v1
---
# Retry Policy
Retry a timeout at most three times.
"""
    meta, _body = parse_document(content, filename="policy.md")

    assert meta.id == "payment-retry-policy-v2"
    assert meta.authority == 100
    assert meta.system == "payments"
    assert meta.supersedes == ["payment-retry-policy-v1"]


def test_malformed_frontmatter_fails_loudly() -> None:
    """SPEC-2.5"""
    with pytest.raises(DocumentValidationError, match="invalid front matter"):
        parse_document(
            "---\nstatus: [broken\n---\nbody",
            filename="broken.md",
        )


def test_policy_resolver_keeps_generic_documents_without_governance_metadata() -> None:
    """SPEC-2.4"""
    generic = metadata(
        doc_id="generic",
        logical_id="generic",
        status=None,
        effective_date=None,
        system=None,
        environment=None,
        document_type=None,
        authority=None,
    )
    governed = metadata(doc_id="governed", logical_id="governed")

    resolved = EvidenceResolver().resolve(
        [evidence(meta=generic, distance=0.1), evidence(meta=governed, distance=0.2)],
        QueryContext(system="payments", environment="production"),
    )

    assert {item.chunk.logical_id for item in resolved} == {"generic", "governed"}
