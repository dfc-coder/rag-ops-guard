from __future__ import annotations

import hashlib
import json

from rag_ops_guard.domain.models import Manifest
from rag_ops_guard.ports import ObjectStore


def document_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def manifest_key(logical_id: str, version: str) -> str:
    return f"manifests/{logical_id}/{version}.json"


def load_manifest(store: ObjectStore, logical_id: str, version: str) -> Manifest | None:
    key = manifest_key(logical_id, version)
    if not store.exists(key):
        return None
    return Manifest.model_validate_json(store.get_text(key))


def save_manifest(store: ObjectStore, manifest: Manifest) -> None:
    store.put_text(
        manifest_key(manifest.logical_id, manifest.version),
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        "application/json",
    )
