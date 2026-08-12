from __future__ import annotations

import re

import yaml
from pydantic import ValidationError

from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import DocumentMetadata

FRONTMATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)
MAX_DOCUMENT_BYTES = 256 * 1024


def parse_document(content: str) -> tuple[DocumentMetadata, str]:
    if len(content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise DocumentValidationError("document exceeds 256 KiB limit")

    match = FRONTMATTER.match(content)
    if match is None:
        raise DocumentValidationError("document must start with YAML front matter")

    try:
        raw = yaml.safe_load(match.group("yaml"))
        if not isinstance(raw, dict):
            raise DocumentValidationError("front matter must be a mapping")
        metadata = DocumentMetadata.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise DocumentValidationError(f"invalid front matter: {exc}") from exc

    body = match.group("body").strip()
    if not body:
        raise DocumentValidationError("document body is empty")
    return metadata, body
