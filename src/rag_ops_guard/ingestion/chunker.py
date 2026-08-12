from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from rag_ops_guard.domain.models import Chunk, DocumentMetadata

_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownChunker:
    def __init__(
        self,
        token_counter: Callable[[str], int],
        target_tokens: int = 400,
        overlap_tokens: int = 60,
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        self._count = token_counter
        self._target = target_tokens
        self._overlap = overlap_tokens

    def split(self, metadata: DocumentMetadata, body: str) -> list[Chunk]:
        sections = self._sections(body)
        chunks: list[Chunk] = []
        carry: list[str] = []
        index = 0

        for header_path, paragraphs in sections:
            current = carry.copy()
            for paragraph in paragraphs:
                candidate = "\n\n".join([*current, paragraph]).strip()
                if current and self._count(candidate) > self._target:
                    chunks.append(self._build(metadata, header_path, index, current))
                    index += 1
                    carry = self._tail(current)
                    current = [*carry, paragraph]
                else:
                    current.append(paragraph)
            if current:
                chunks.append(self._build(metadata, header_path, index, current))
                index += 1
                carry = self._tail(current)

        return chunks

    def _sections(self, body: str) -> list[tuple[list[str], list[str]]]:
        header_stack: list[str] = []
        sections: list[tuple[list[str], list[str]]] = []
        paragraphs: list[str] = []
        buffer: list[str] = []

        def flush_paragraph() -> None:
            if buffer:
                paragraphs.append("\n".join(buffer).strip())
                buffer.clear()

        def flush_section() -> None:
            flush_paragraph()
            if paragraphs:
                sections.append((header_stack.copy(), paragraphs.copy()))
                paragraphs.clear()

        for line in body.splitlines():
            header = _HEADER.match(line)
            if header:
                flush_section()
                level = len(header.group(1))
                title = header.group(2)
                header_stack[:] = header_stack[: level - 1]
                header_stack.append(title)
            elif not line.strip():
                flush_paragraph()
            else:
                buffer.append(line)
        flush_section()

        if not sections and body.strip():
            sections.append(([], [body.strip()]))
        return sections

    def _tail(self, paragraphs: list[str]) -> list[str]:
        tail: list[str] = []
        for paragraph in reversed(paragraphs):
            candidate = [paragraph, *tail]
            if self._count("\n\n".join(candidate)) > self._overlap:
                break
            tail = candidate
        return tail

    @staticmethod
    def _build(
        metadata: DocumentMetadata,
        header_path: list[str],
        index: int,
        paragraphs: list[str],
    ) -> Chunk:
        text = "\n\n".join(paragraphs).strip()
        digest = hashlib.sha256(text.encode()).hexdigest()
        chunk_id = f"{metadata.logical_id}:{metadata.version}:{index:03d}:{digest[:8]}"
        return Chunk(
            id=chunk_id,
            logical_id=metadata.logical_id,
            version=metadata.version,
            title=metadata.title,
            header_path=header_path,
            chunk_index=index,
            text=text,
            metadata=metadata,
            content_sha256=digest,
        )
