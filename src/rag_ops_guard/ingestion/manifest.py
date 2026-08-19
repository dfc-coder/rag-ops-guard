from __future__ import annotations

import hashlib
import json

from rag_ops_guard.domain.models import Manifest
from rag_ops_guard.ports import ObjectStore
from rag_ops_guard.tenancy import KeyLayout

_DEFAULT_LAYOUT = KeyLayout("default")


def document_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def manifest_key(logical_id: str, version: str, layout: KeyLayout | None = None) -> str:
    return (layout or _DEFAULT_LAYOUT).manifest_key(logical_id, version)


def load_manifest(
    store: ObjectStore,
    logical_id: str,
    version: str,
    layout: KeyLayout | None = None,
) -> Manifest | None:
    key = manifest_key(logical_id, version, layout)
    if not store.exists(key):
        return None
    return Manifest.model_validate_json(store.get_text(key))


def save_manifest(
    store: ObjectStore,
    manifest: Manifest,
    layout: KeyLayout | None = None,
) -> None:
    store.put_text(
        manifest_key(manifest.logical_id, manifest.version, layout),
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        "application/json",
    )
