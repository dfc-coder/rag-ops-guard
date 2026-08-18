from __future__ import annotations

import shutil
import tempfile
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


def _validated_candidate(text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".env", delete=False) as handle:
        handle.write(text)
        candidate = Path(handle.name)
    try:
        Settings(_env_file=candidate)
    finally:
        candidate.unlink(missing_ok=True)


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

    # Validate the complete candidate before touching the developer's real file.
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
