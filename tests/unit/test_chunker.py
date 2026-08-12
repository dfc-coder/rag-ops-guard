from rag_ops_guard.ingestion.chunker import MarkdownChunker
from tests.fixtures.builders import metadata


def words(text: str) -> int:
    return len(text.split())


def test_chunker_keeps_header_path_and_stable_ids() -> None:
    body = """# Timeout Handling
First paragraph has several words about retry behavior.

Second paragraph contains additional operational guidance and validation steps.
"""
    chunker = MarkdownChunker(words, target_tokens=12, overlap_tokens=3)
    first = chunker.split(metadata(), body)
    second = chunker.split(metadata(), body)
    assert [item.id for item in first] == [item.id for item in second]
    assert all(item.header_path == ["Timeout Handling"] for item in first)
    assert len(first) >= 2


def test_overlap_must_be_smaller_than_target() -> None:
    try:
        MarkdownChunker(words, target_tokens=10, overlap_tokens=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")
