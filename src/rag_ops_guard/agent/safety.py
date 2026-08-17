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

_SECRET_PHRASES = (
    "api key",
    "secret token",
    "authentication data",
    "authentication secret",
    "production credentials",
    "clave de api",
    "clave api",
    "datos de autenticacion",
    "datos de autenticación",
    "credenciales de produccion",
    "credenciales de producción",
)

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

_SAFE_SECRET_CONTEXT_TERMS = {
    "rotate",
    "rotating",
    "rotation",
    "policy",
    "policies",
    "manage",
    "management",
    "store",
    "storage",
    "protect",
    "protection",
    "format",
    "example",
    "examples",
    "configure",
    "configuration",
    "rotar",
    "rotacion",
    "rotación",
    "politica",
    "política",
    "politicas",
    "políticas",
    "gestionar",
    "gestion",
    "gestión",
    "proteger",
    "ejemplo",
    "ejemplos",
    "configurar",
    "configuracion",
    "configuración",
}

_DIRECT_SECRET_PATTERNS = (
    re.compile(r"\b(?:what|which)\s+(?:is|are)\s+the\b.*\b(?:api key|password|secret token|credentials?)\b"),
    re.compile(r"\b(?:cu[aá]l|cu[aá]les)\s+(?:es|son)\s+(?:la|el|las|los)\b.*\b(?:clave (?:de )?api|contrase(?:ñ|n)a|token secreto|credenciales?)\b"),
)

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
        contains_secret = bool(tokens.intersection(_SECRET_TERMS)) or any(
            phrase in normalized for phrase in _SECRET_PHRASES
        )
        asks_for_secret = contains_secret and bool(tokens.intersection(_EXTRACTION_TERMS))
        direct_secret = (
            contains_secret
            and not tokens.intersection(_SAFE_SECRET_CONTEXT_TERMS)
            and any(pattern.search(normalized) for pattern in _DIRECT_SECRET_PATTERNS)
        )
        bypass = any(pattern in normalized for pattern in _BYPASS_PATTERNS)
        return asks_for_secret or direct_secret or bypass
