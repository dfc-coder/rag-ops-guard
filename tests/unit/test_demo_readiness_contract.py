from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_ops_guard.agent.responses import (
    capabilities_response,
    insufficient_evidence_response,
    out_of_scope_response,
    runtime_error_response,
)
from rag_ops_guard.domain.models import QueryRequest


@pytest.mark.parametrize("message", ["ok", "si", "no", "hi", "a"])
def test_short_conversational_messages_are_valid(message: str) -> None:
    request = QueryRequest(question=message)
    assert request.question == message


def test_question_is_trimmed() -> None:
    request = QueryRequest(question="  hola  ")
    assert request.question == "hola"


def test_whitespace_only_question_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


@pytest.mark.parametrize(
    ("question", "renderer"),
    [
        ("dime que sabes hacer", capabilities_response),
        ("algo fuera de alcance", out_of_scope_response),
        ("no encuentro ese dato", insufficient_evidence_response),
        ("fallo inesperado", runtime_error_response),
    ],
)
def test_spanish_control_responses_default_to_spanish_without_accents(
    question: str,
    renderer,
) -> None:
    text = renderer(question).lower()
    assert "i can" not in text
    assert "available documentation" not in text
    assert "could not" not in text
    assert "unexpected error" not in text


def test_clear_english_control_question_stays_english() -> None:
    assert capabilities_response("What can you do?").startswith("I can")
