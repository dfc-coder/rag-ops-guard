from __future__ import annotations

from collections import defaultdict
from datetime import date

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
            group.sort(key=self._precedence, reverse=True)
            selected.extend(self._without_superseded(group))

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
        if context.api_version and meta.version != context.api_version:
            return False
        return True

    @staticmethod
    def _precedence(item: Evidence) -> tuple[date, int, tuple[int, ...]]:
        meta = item.chunk.metadata
        version = tuple(int(part) if part.isdigit() else 0 for part in meta.version.split("."))
        return (meta.effective_date, meta.authority, version)

    @staticmethod
    def _without_superseded(group: list[Evidence]) -> list[Evidence]:
        superseded = {
            superseded_id
            for item in group
            for superseded_id in item.chunk.metadata.supersedes
        }
        return [item for item in group if item.chunk.metadata.id not in superseded]
