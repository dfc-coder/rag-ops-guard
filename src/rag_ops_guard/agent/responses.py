from __future__ import annotations

import re

_SPANISH_MARKERS = {
    "que",
    "qué",
    "puedes",
    "podés",
    "tienes",
    "tenés",
    "documentacion",
    "documentación",
    "cual",
    "cuál",
    "gracias",
    "hola",
}


def capabilities_response(question: str) -> str:
    if _is_spanish(question):
        return (
            "Puedo conversar sobre operaciones de integración, consultar la knowledge base cuando "
            "necesito evidencia, listar la documentación disponible, mantener el contexto de "
            "follow-ups y responder con citas a las fuentes utilizadas."
        )
    return (
        "I can discuss integration operations, search the knowledge base when evidence is needed, "
        "list the available documentation, keep context across follow-ups, and answer with citations "
        "to the sources used."
    )


def out_of_scope_response(question: str) -> str:
    if _is_spanish(question):
        return (
            "Estoy enfocado en operaciones de integración y en la documentación disponible en esta "
            "knowledge base. Puedo ayudarte con APIs, runbooks, incidentes, SLAs, arquitectura y "
            "políticas operativas documentadas."
        )
    return (
        "I am focused on integration operations and the documentation available in this knowledge "
        "base. I can help with documented APIs, runbooks, incidents, SLAs, architecture, and "
        "operational policies."
    )


def insufficient_evidence_response(question: str) -> str:
    if _is_spanish(question):
        return "No encontré evidencia suficientemente relevante en la documentación disponible."
    return "I could not find sufficiently relevant evidence in the available documentation."


def _is_spanish(text: str) -> bool:
    tokens = {match.group(0).casefold() for match in re.finditer(r"\w+", text, re.UNICODE)}
    return bool(tokens.intersection(_SPANISH_MARKERS)) or any(char in text for char in "¿¡áéíóúñ")
