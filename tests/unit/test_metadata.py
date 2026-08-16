import pytest

from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.ingestion.metadata import parse_document

VALID = """---
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


def test_parse_document_returns_metadata_and_body() -> None:
    metadata, body = parse_document(VALID)
    assert metadata.logical_id == "payment-retry-policy"
    assert metadata.version == "2.0"
    assert "three times" in body


def test_parse_document_without_frontmatter_uses_heading() -> None:
    """SPEC-2.1"""
    metadata, body = parse_document("# Missing metadata\n\nGeneric body")
    assert metadata.title == "Missing metadata"
    assert metadata.authority is None
    assert body.endswith("Generic body")


def test_parse_document_rejects_empty_body() -> None:
    with pytest.raises(DocumentValidationError, match="body is empty"):
        parse_document(VALID.split("# Retry Policy")[0])


def test_parse_document_rejects_oversized_content() -> None:
    oversized = VALID + ("x" * (256 * 1024))
    with pytest.raises(DocumentValidationError, match="256 KiB"):
        parse_document(oversized)
