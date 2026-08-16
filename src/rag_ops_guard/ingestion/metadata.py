from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import DocumentMetadata

FRONTMATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL)
HEADING = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
MAX_DOCUMENT_BYTES = 256 * 1024
SUPPORTED_SUFFIXES = {".md", ".txt"}


def parse_document(content: str, filename: str | None = None) -> tuple[DocumentMetadata, str]:
    if len(content.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise DocumentValidationError("document exceeds 256 KiB limit")

    _validate_format(filename)
    match = FRONTMATTER.match(content)

    if match is None:
        if content.startswith("---"):
            raise DocumentValidationError("invalid front matter: missing or malformed delimiter")
        body = content.strip()
        if not body:
            raise DocumentValidationError("document body is empty")
        return DocumentMetadata(**_generic_identity(content, body, filename)), body

    body = match.group("body").strip()
    if not body:
        raise DocumentValidationError("document body is empty")

    try:
        raw = yaml.safe_load(match.group("yaml"))
        if not isinstance(raw, dict):
            raise DocumentValidationError("invalid front matter: expected a mapping")
        identity = _generic_identity(content, body, filename)
        for key, value in identity.items():
            raw.setdefault(key, value)
        metadata = DocumentMetadata.model_validate(raw)
    except DocumentValidationError:
        raise
    except (yaml.YAMLError, ValidationError) as exc:
        raise DocumentValidationError(f"invalid front matter: {exc}") from exc

    return metadata, body


def _validate_format(filename: str | None) -> None:
    if filename is None:
        return
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        formats = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentValidationError(f"supported formats are {formats}")


def _generic_identity(content: str, body: str, filename: str | None) -> dict[str, str]:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    document_id = f"doc-{digest[:16]}"
    return {
        "id": document_id,
        "logical_id": document_id,
        "title": _infer_title(body, filename),
        "version": "1",
    }


def _infer_title(body: str, filename: str | None) -> str:
    heading = HEADING.search(body)
    if heading:
        return heading.group(1).strip()

    if filename:
        stem = Path(filename).stem
        title = re.sub(r"[-_]+", " ", stem).strip()
        if title:
            return title

    return "Untitled document"
