from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_SAFE_PART = re.compile(r"^[A-Za-z0-9._@+:-]+$")


@dataclass(frozen=True, slots=True)
class KeyLayout:
    """Canonical tenant-scoped storage and vector naming contract."""

    tenant_id: str

    def __post_init__(self) -> None:
        if not _TENANT_ID.fullmatch(self.tenant_id):
            raise ValueError("tenant_id must be a safe lowercase identifier")

    @property
    def tenant_prefix(self) -> str:
        return f"t/{self.tenant_id}/"

    @property
    def raw_prefix(self) -> str:
        return f"{self.tenant_prefix}raw/"

    @property
    def chunks_prefix(self) -> str:
        return f"{self.tenant_prefix}chunks/"

    @property
    def manifests_prefix(self) -> str:
        return f"{self.tenant_prefix}manifests/"

    def raw_key(self, relative_key: str) -> str:
        return f"{self.raw_prefix}{self._relative_path(relative_key)}"

    def chunk_prefix(self, logical_id: str, version: str) -> str:
        return f"{self.chunks_prefix}{self._part(logical_id)}/{self._part(version)}/"

    def chunk_key(self, logical_id: str, version: str, chunk_index: int) -> str:
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        return f"{self.chunk_prefix(logical_id, version)}chunk-{chunk_index:03d}.json"

    def manifest_key(self, logical_id: str, version: str) -> str:
        return f"{self.manifests_prefix}{self._part(logical_id)}/{self._part(version)}.json"

    def owns(self, key: str) -> bool:
        return key.startswith(self.tenant_prefix)

    def require_ingest_key(self, key: str) -> str:
        if not key.startswith(self.raw_prefix):
            raise ValueError("ingestion requires a tenant-scoped raw key")
        relative = key[len(self.raw_prefix) :]
        self._relative_path(relative)
        return key

    def vector_index(self, base_index: str) -> str:
        base = self._part(base_index)
        return f"{base}--{self.tenant_id}"

    @staticmethod
    def _part(value: str) -> str:
        if not value or not _SAFE_PART.fullmatch(value):
            raise ValueError("storage identifier contains unsafe characters")
        return value

    @staticmethod
    def _relative_path(value: str) -> str:
        if not value or value.startswith("/"):
            raise ValueError("relative storage path is required")
        path = PurePosixPath(value)
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative storage path contains unsafe traversal")
        normalized = str(path)
        if normalized != value:
            raise ValueError("relative storage path must be normalized")
        return normalized
