from __future__ import annotations

from dataclasses import dataclass

from rag_ops_guard.agent.responses import _is_spanish
from rag_ops_guard.domain.models import Chunk, DocumentStatus, QueryContext
from rag_ops_guard.ports import ObjectStore


@dataclass(frozen=True)
class CatalogEntry:
    title: str
    document_type: str
    system: str
    version: str


class KnowledgeCatalog:
    def __init__(self, objects: ObjectStore) -> None:
        self._objects = objects

    def entries(self, context: QueryContext | None = None) -> list[CatalogEntry]:
        context = context or QueryContext()
        seen: set[tuple[str, str]] = set()
        entries: list[CatalogEntry] = []
        for key in self._objects.list_keys("chunks/"):
            if not key.endswith(".json"):
                continue
            chunk = Chunk.model_validate_json(self._objects.get_text(key))
            meta = chunk.metadata
            if meta.status is not None and meta.status != DocumentStatus.ACTIVE:
                continue
            if context.system and meta.system is not None and meta.system != context.system:
                continue
            if (
                context.environment
                and meta.environment is not None
                and meta.environment not in {context.environment, "all"}
            ):
                continue
            identity = (meta.logical_id, meta.version)
            if identity in seen:
                continue
            seen.add(identity)
            entries.append(
                CatalogEntry(
                    title=meta.title,
                    document_type=meta.document_type.value if meta.document_type else "document",
                    system=meta.system or "general",
                    version=meta.version,
                )
            )
        return sorted(entries, key=lambda item: (item.document_type, item.title.casefold()))

    def render(self, question: str, context: QueryContext | None = None) -> str:
        entries = self.entries(context)
        if not entries:
            if _is_spanish(question):
                return "No hay documentación disponible para ese contexto."
            return "There is no documentation available for that context."

        groups: dict[str, list[CatalogEntry]] = {}
        for entry in entries:
            groups.setdefault(entry.document_type, []).append(entry)

        if _is_spanish(question):
            header = "Documentación disponible:"
        else:
            header = "Available documentation:"

        lines = [header]
        for document_type in sorted(groups):
            lines.append(f"\n**{document_type}**")
            for entry in groups[document_type]:
                lines.append(f"- {entry.title} · v{entry.version} · {entry.system}")
        return "\n".join(lines)
