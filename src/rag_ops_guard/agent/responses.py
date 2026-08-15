from __future__ import annotations

import re

_ENGLISH_MARKERS = {
    "what",
    "which",
    "who",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "are",
    "is",
    "the",
    "available",
    "documentation",
    "documents",
    "help",
    "hello",
    "thanks",
    "ignore",
    "give",
    "production",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "bypass",
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
        "list available documentation, keep context across follow-ups, and answer with "
        "citations to the sources used."
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


def safety_blocked_response(question: str) -> str:
    if _is_spanish(question):
        return "No puedo ayudar a extraer secretos ni a omitir controles operativos o de seguridad."
    return "I cannot help extract secrets or bypass operational or security controls."


def runtime_error_response(question: str) -> str:
    if _is_spanish(question):
        return "No pude completar esa consulta. El servicio sigue disponible; probá nuevamente."
    return "I could not complete that request. The service is still available; please try again."


def _is_spanish(text: str) -> bool:
    """Prefer Spanish unless the user message contains a clear English signal.

    Demo traffic is primarily Spanish and many valid Spanish turns contain no accents. A small
    positive-English detector is safer for deterministic control messages than attempting to infer
    Spanish from a narrow list of accented/functional words.
    """
    tokens = {match.group(0).casefold() for match in re.finditer(r"\w+", text, re.UNICODE)}
    if any(char in text for char in "¿¡áéíóúñ"):
        return True
    return not bool(tokens.intersection(_ENGLISH_MARKERS))