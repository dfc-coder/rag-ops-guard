from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag_ops_guard.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_settings_from_developer_dotenv() -> Iterator[None]:
    """Unit/integration tests must not depend on a developer's checkout .env.

    Explicit `_env_file=...` arguments used by configuration-contract tests still
    take precedence, while ordinary Settings()/get_settings() calls read only
    process environment during tests.
    """

    original_env_file = Settings.model_config.get("env_file")
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
        get_settings.cache_clear()
