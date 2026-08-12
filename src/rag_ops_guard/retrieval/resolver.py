from __future__ import annotations

from collections import defaultdict
from datetime import date

from rag_ops_guard.domain.errors import EvidenceConflictError
from rag_ops_guard.domain.models import DocumentStatus, Evidence, QueryContext


class EvidenceResolver:
    def resolve(
        self, evidence: list[Evidence], context: QueryContext, limit: int = 5
    ) -> list[Evidence]:
        eligible = [item for item in evidence if self._eligible(item, context)]
        grouped: dict[str, list[Evidence]] = defaultdict(list)
        for item in eligible:
            grouped[item.chunk.logical_id].append(item)

        selected: list[Evidence] = []
        for group in grouped.values():
            selected.extend(self._resolve_logical_document(group))

        selected.sort(key=lambda item: (item.distance is None, item.distance or 0.0))
        return selected[:limit]

    @staticmethod
    def _eligible(item: Evidence, context: QueryContext) -> bool:
        meta = item.chunk.metadata
        if meta.status != DocumentStatus.ACTIVE:
            return False
        if context.system and meta.system != context.system:
            return False
        if context.environment and meta.environment not in {context.environment, "all"}:
            return False
        return not context.api_version or meta.version == context.api_version

    def _resolve_logical_document(self, group: list[Evidence]) -> list[Evidence]:
        by_document: dict[str, list[Evidence]] = defaultdict(list)
        for item in group:
            by_document[item.chunk.metadata.id].append(item)

        superseded = {
            superseded_id
            for item in group
            for superseded_id in item.chunk.metadata.supersedes
        }
        candidates = {
            document_id: items
            for document_id, items in by_document.items()
            if document_id not in superseded
        }
        if not candidates:
            return []

        precedence_by_document = {
            document_id: self._precedence(items[0])
            for document_id, items in candidates.items()
        }
        highest = max(precedence_by_document.values())
        winners = [
            document_id
            for document_id, precedence in precedence_by_document.items()
            if precedence == highest
        ]
        if len(winners) > 1:
            raise EvidenceConflictError(
                "equally authoritative active documents cannot be resolved: "
                + ", ".join(sorted(winners))
            )
        return candidates[winners[0]]

    @staticmethod
    def _precedence(item: Evidence) -> tuple[date, int, tuple[int, ...]]:
        meta = item.chunk.metadata
        version = tuple(int(part) if part.isdigit() else 0 for part in meta.version.split("."))
        return (meta.effective_date, meta.authority, version)
