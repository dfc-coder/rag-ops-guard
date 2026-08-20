from __future__ import annotations

import os

from rag_ops_guard.config import Settings


def evaluation_settings_from_environment() -> Settings:
    """Build explicit evaluation settings at the tooling boundary.

    Evaluation commands are operator tooling, not application runtime code. They may consume
    process environment variables and pass a fully validated Settings object into the package.
    """

    values: dict[str, object] = {}
    for field_name in Settings.model_fields:
        environment_name = field_name.upper()
        if environment_name in os.environ:
            values[field_name] = os.environ[environment_name]
    return Settings.model_validate(values)
