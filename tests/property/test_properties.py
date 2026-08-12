from contextlib import suppress
from datetime import date

import pytest
from hypothesis import given, strategies as st

from rag_ops_guard.domain.errors import CitationValidationError, DocumentValidationError
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ingestion.metadata import parse_document
from rag_ops_guard.retrieval.citations import validate_citations
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence, metadata


@given(st.text(max_size=2000))
def test_parser_never_leaks_unexpected_exception(text: str) -> None:
    with suppress(DocumentValidationError):
        parse_document(text)


@given(st.permutations([0, 1, 2]))
def test_resolver_is_deterministic_for_input_order(order: list[int]) -> None:
    items = [
        evidence(
            meta=metadata(doc_id="a", logical_id="a", effective_date=date(2026, 1, 1)),
            distance=0.3,
        ),
        evidence(
            meta=metadata(doc_id="b", logical_id="b", effective_date=date(2026, 2, 1)),
            distance=0.1,
        ),
        evidence(
            meta=metadata(doc_id="c", logical_id="c", effective_date=date(2026, 3, 1)),
            distance=0.2,
        ),
    ]
    resolver = EvidenceResolver()
    expected = [item.chunk.id for item in resolver.resolve(items, QueryContext())]
    actual = [
        item.chunk.id
        for item in resolver.resolve([items[i] for i in order], QueryContext())
    ]
    assert actual == expected


@given(
    st.text(min_size=1, max_size=80).filter(
        lambda value: value != "payment-retry-policy:2.0:000:deadbeef"
    )
)
def test_unknown_citation_ids_are_never_accepted(citation_id: str) -> None:
    with pytest.raises(CitationValidationError):
        validate_citations([citation_id], [evidence()])
