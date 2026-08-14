from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\wáéíóúñ]+", re.UNICODE | re.IGNORECASE)

_SECRET_TERMS = {
    "credential",
    "credentials",
    "password",
    "passwords",
    "secret",
    "secrets",
    "token",
    "tokens",
    "api_key",
    "apikey",
    "authentication",
    "auth",
    "credencial",
    "credenciales",
    "contraseña",
    "contrasena",
    "contraseñas",
    "contrasenas",
    "secreto",
    "secretos",
    "autenticación",
    "autenticacion",
}

_EXTRACTION_TERMS = {
    "reveal",
    "expose",
    "show",
    "give",
    "provide",
    "print",
    "dump",
    "display",
    "mostrar",
    "muestra",
    "dame",
    "dar",
    "revelar",
    "revela",
    "exponer",
    "expone",
    "proporcionar",
    "proporciona",
    "pasame",
    "pásame",
}

_BYPASS_PATTERNS = (
    "ignore policy",
    "ignore policies",
    "ignore controls",
    "bypass policy",
    "bypass the policy",
    "bypass the runbook",
    "bypass controls",
    "skip security controls",
    "ignora la politica",
    "ignora la política",
    "ignora las politicas",
    "ignora las políticas",
    "ignora los controles",
    "evita los controles",
    "omiti la politica",
    "omití la política",
    "saltea la politica",
    "salteá la política",
)


class SafetyGuard:
    """Deterministic guard for explicit secret extraction or policy-bypass requests."""

    def blocked(self, text: str) -> bool:
        normalized = " ".join(text.casefold().split())
        tokens = {match.group(0).casefold() for match in _TOKEN_RE.finditer(normalized)}
        asks_for_secret = bool(tokens.intersection(_SECRET_TERMS)) and bool(
            tokens.intersection(_EXTRACTION_TERMS)
        )
        bypass = any(pattern in normalized for pattern in _BYPASS_PATTERNS)
        return asks_for_secret or bypass
