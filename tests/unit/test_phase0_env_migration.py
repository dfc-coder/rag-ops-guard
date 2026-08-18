from __future__ import annotations

from pathlib import Path

from scripts.local.migrate_phase0_env import migrate, migrated_text


def test_migration_removes_only_known_deprecated_keys() -> None:
    source = (
        "APP_ENV=local\n"
        "ROUTER_MIN_SCORE=0.35\n"
        "LLM_MODEL=qwen3.5-2b-unsloth-ud-q4-k-xl\n"
        "RETRIEVAL_RELEVANCE_THRESHOLD=0.4\n"
        "ROUTER_MIN_MARGIN=0.015\n"
    )

    result, removed = migrated_text(source)

    assert removed == [
        "RETRIEVAL_RELEVANCE_THRESHOLD",
        "ROUTER_MIN_MARGIN",
        "ROUTER_MIN_SCORE",
    ]
    assert result == (
        "APP_ENV=local\n"
        "LLM_MODEL=qwen3.5-2b-unsloth-ud-q4-k-xl\n"
    )


def test_migration_backs_up_before_writing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_ENV=local\n"
        "ROUTER_MIN_SCORE=0.35\n",
        encoding="utf-8",
    )

    assert migrate(env_file) == 0

    assert env_file.read_text(encoding="utf-8") == "APP_ENV=local\n"
    backup = tmp_path / ".env.pre-phase0.bak"
    assert "ROUTER_MIN_SCORE=0.35" in backup.read_text(encoding="utf-8")
