from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin

from pydantic import SecretStr
from pydantic_core import PydanticUndefined

from rag_ops_guard.config import Settings


class ConfigScope(str, Enum):
    REGISTRY = "REGISTRY"
    GLOBAL = "GLOBAL"


@dataclass(frozen=True)
class RegistryEntry:
    name: str
    value_type: str
    json_schema: dict[str, Any]
    sensitivity: str
    scope: str
    fail_mode: str
    reload_policy: str
    code_default: object | None


_FAIL_CLOSED_KEYS = {
    "retrieval_domain_min_relevance",
    "retrieval_min_relevance",
    "llm_base_url",
    "embedding_base_url",
    "reranker_base_url",
}


def _contains_secret(annotation: object) -> bool:
    if annotation is SecretStr:
        return True
    origin = get_origin(annotation)
    if origin is None and not isinstance(annotation, UnionType):
        return False
    return any(_contains_secret(arg) for arg in get_args(annotation))


def _value_type(schema: dict[str, Any]) -> str:
    if "type" in schema:
        return str(schema["type"])
    for branch in schema.get("anyOf", []):
        if branch.get("type") != "null" and "type" in branch:
            return str(branch["type"])
    return "object"


def registry_entries() -> dict[str, RegistryEntry]:
    schema = Settings.model_json_schema().get("properties", {})
    entries: dict[str, RegistryEntry] = {}
    for name, field in Settings.model_fields.items():
        secret = _contains_secret(field.annotation)
        field_schema = dict(schema.get(name, {}))
        default: object | None
        if secret or field.default is PydanticUndefined:
            default = None
        else:
            default = field.default
        entries[name] = RegistryEntry(
            name=name,
            value_type=_value_type(field_schema),
            json_schema=field_schema,
            sensitivity="secret" if secret else "internal",
            scope="global_only",
            fail_mode="fail_closed"
            if secret or name in _FAIL_CLOSED_KEYS
            else "fail_open_stale",
            reload_policy="on_revision",
            code_default=default,
        )
    return entries


def public_settings_values(settings: Settings) -> dict[str, object]:
    entries = registry_entries()
    return {
        name: getattr(settings, name)
        for name, entry in entries.items()
        if entry.sensitivity != "secret"
    }


def validate_value_against_schema(entry: RegistryEntry, value: object) -> None:
    """Validate the JSON-schema constraints emitted by Settings without a new dependency."""
    schema = entry.json_schema
    expected = schema.get("type")
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{entry.name}: expected integer")
    if expected == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        raise ValueError(f"{entry.name}: expected number")
    if expected == "boolean" and not isinstance(value, bool):
        raise ValueError(f"{entry.name}: expected boolean")
    if expected == "string" and not isinstance(value, str):
        raise ValueError(f"{entry.name}: expected string")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{entry.name}: value outside enum")
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and isinstance(value, (int, float)) and value < minimum:
        raise ValueError(f"{entry.name}: below minimum {minimum}")
    if maximum is not None and isinstance(value, (int, float)) and value > maximum:
        raise ValueError(f"{entry.name}: above maximum {maximum}")
