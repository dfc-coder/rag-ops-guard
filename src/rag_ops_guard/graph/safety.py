from __future__ import annotations

import re
from typing import Literal

SafetyCategory = Literal["normal", "secret_extraction", "policy_bypass"]

_SECRET_REQUEST = re.compile(
    r"\b(?:reveal|expose|extract|print|show|give|return|provide|what\s+is|tell\s+me)\b"
    r".{0,60}\b(?:credential|credentials|password|api\s*key|secret(?:\s+token)?|access\s+token)\b",
    re.IGNORECASE,
)
_POLICY_BYPASS = re.compile(
    r"\b(?:ignore|bypass|circumvent|override|disregard|skip)\b"
    r".{0,60}\b(?:policy|policies|runbook|instructions?|guardrails?|restrictions?)\b",
    re.IGNORECASE,
)


def classify_direct_safety(question: str) -> SafetyCategory:
    """Classify only explicit direct attacks; operational policy questions remain normal."""
    if _SECRET_REQUEST.search(question):
        return "secret_extraction"
    if _POLICY_BYPASS.search(question):
        return "policy_bypass"
    return "normal"
