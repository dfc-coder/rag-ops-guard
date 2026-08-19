"""Transitional Phase 6.1 adapter from process environment to explicit Settings values.

``Settings`` itself is environment-agnostic. This adapter preserves the existing
Phase 5/6.0 runtime contract until Phase 6.2 replaces process-environment reads
with AppConfig/control-plane inputs.
"""

from __future__ import annotations

import os

from rag_ops_guard.config import Settings


def settings_from_process_environment() -> Settings:
    """Build ``Settings`` from explicitly collected known environment fields."""

    values: dict[str, object] = {}
    for field_name in Settings.model_fields:
        environment_name = field_name.upper()
        if environment_name in os.environ:
            values[field_name] = os.environ[environment_name]
    return Settings.model_validate(values)
