from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import SecretStr


def _canonical_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return hashlib.sha256(value.get_secret_value().encode("utf-8")).hexdigest()
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_config_json(values: Mapping[str, object]) -> bytes:
    """Return the machine-independent canonical JSON representation from INV-8."""
    canonical = {str(key): _canonical_value(value) for key, value in values.items()}
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_hash(values: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_config_json(values)).hexdigest()
