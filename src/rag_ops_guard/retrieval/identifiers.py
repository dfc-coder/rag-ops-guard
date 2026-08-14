from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[^\W\d_][\w.-]*", re.UNICODE)
_NON_IDENTIFIER_WORDS = {
    "and",
    "como",
    "cómo",
    "cual",
    "cuál",
    "cuando",
    "cuándo",
    "dime",
    "decime",
    "donde",
    "dónde",
    "explain",
    "how",
    "que",
    "qué",
    "quien",
    "quién",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "y",
}


def explicit_identifiers(text: str) -> set[str]:
    """Extract explicit named/technical identifiers without an entity allowlist."""
    identifiers: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        normalized = token.casefold()
        if normalized in _NON_IDENTIFIER_WORDS or len(token) < 2:
            continue

        has_upper = any(char.isupper() for char in token)
        has_lower = any(char.islower() for char in token)
        is_acronym = token.isupper() and any(char.isalpha() for char in token)
        is_camel_case = has_upper and has_lower and any(char.isupper() for char in token[1:])
        is_title_case = token[:1].isupper() and token[1:].islower()
        has_digit = any(char.isdigit() for char in token)

        if is_acronym or is_camel_case or is_title_case or has_digit:
            identifiers.add(normalized)
    return identifiers


def text_tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN_RE.finditer(text)}


def identifiers_match_text(identifiers: set[str], text: str) -> bool:
    """No identifier constraint means no filtering; otherwise require one exact anchor."""
    if not identifiers:
        return True
    return bool(identifiers.intersection(text_tokens(text)))


def focus_allows_rewrite(
    current_question: str,
    previous_query: str,
    source_titles: list[str],
) -> bool:
    """Prevent trusted memory from crossing an explicit named-topic boundary."""
    current_identifiers = explicit_identifiers(current_question)
    if not current_identifiers:
        return True

    focus_text = "\n".join([previous_query, *source_titles])
    return identifiers_match_text(current_identifiers, focus_text)
