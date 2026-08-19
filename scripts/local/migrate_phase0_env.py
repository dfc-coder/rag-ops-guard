from __future__ import annotations

import shutil
from pathlib import Path

from rag_ops_guard.config import Settings

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
DEPRECATED_KEYS = {
    "RETRIEVAL_RELEVANCE_THRESHOLD",
    "ROUTER_MIN_SCORE",
    "ROUTER_MIN_MARGIN",
}


def _key_of(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line.removeprefix("export ").strip()
    return line.split("=", 1)[0].strip().upper()


def migrated_text(text: str) -> tuple[str, list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for raw_line in text.splitlines(keepends=True):
        key = _key_of(raw_line)
        if key in DEPRECATED_KEYS:
            removed.append(key)
            continue
        kept.append(raw_line)
    return "".join(kept), sorted(set(removed))


def _explicit_settings_values(text: str) -> dict[str, object]:
    fields_by_environment_name = {name.upper(): name for name in Settings.model_fields}
    values: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        raw_key, raw_value = line.split("=", 1)
        field_name = fields_by_environment_name.get(raw_key.strip().upper())
        if field_name is None:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[field_name] = value
    return values


def _validated_candidate(text: str) -> None:
    Settings.model_validate(_explicit_settings_values(text))


def migrate(path: Path = ENV_FILE) -> int:
    if not path.is_file():
        print(f"No .env found at {path}; nothing to migrate")
        return 0

    original = path.read_text(encoding="utf-8")
    candidate, removed = migrated_text(original)
    if not removed:
        _validated_candidate(original)
        print("Phase-0 .env migration: no deprecated keys found; configuration is valid")
        return 0

    _validated_candidate(candidate)

    backup = path.with_name(f"{path.name}.pre-phase0.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(candidate, encoding="utf-8")

    print("Phase-0 .env migration complete")
    print(f"Backup: {backup}")
    print("Removed deprecated keys: " + ", ".join(removed))
    return 0


if __name__ == "__main__":
    raise SystemExit(migrate())
