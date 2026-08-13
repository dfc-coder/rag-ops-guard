from __future__ import annotations

import argparse
import html
import re
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rag_ops_guard.app import ingestion_service, object_store, query_workflow
from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.ingestion.metadata import MAX_DOCUMENT_BYTES, parse_document

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge-base"
WEB_DIR = ROOT / "web"
TEMPLATE = (WEB_DIR / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="RAG Ops Guard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
SESSION_UPLOADS: list[dict[str, str]] = []

STATUS_LABELS = {
    "answered": "Respondido",
    "insufficient_evidence": "Evidencia insuficiente",
    "clarification_required": "Necesita aclaración",
    "safety_blocked": "Bloqueado por seguridad",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "document"


def _ingest(key: str, content: str) -> dict[str, object]:
    object_store().put_text(key, content, "text/markdown")
    return ingestion_service().ingest(key).model_dump(mode="json")


def _prepare_upload(
    file: UploadFile,
    title: str,
    system: str,
    environment: str,
    document_type: str,
    version: str,
    authority: int,
) -> tuple[dict[str, object], dict[str, str]]:
    filename = Path(file.filename or "document.md").name
    if Path(filename).suffix.lower() not in {".md", ".markdown"}:
        raise HTTPException(415, "Beta 1 supports Markdown files only")
    raw = file.file.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise HTTPException(413, "Document exceeds the 256 KiB limit")
    try:
        content = raw.decode("utf-8")
        if not content.lstrip().startswith("---"):
            resolved_title = title.strip() or Path(filename).stem.replace("-", " ").title()
            logical_id = _slug(resolved_title)
            metadata = {
                "id": f"{logical_id}-v{_slug(version)}",
                "logical_id": logical_id,
                "title": resolved_title,
                "version": version or "1.0",
                "status": "active",
                "effective_date": date.today().isoformat(),
                "system": system or "general",
                "environment": environment,
                "document_type": document_type,
                "authority": authority,
                "supersedes": [],
            }
            content = f"---\n{yaml.safe_dump(metadata, sort_keys=False).strip()}\n---\n\n{content}"
        parsed, _ = parse_document(content)
    except (UnicodeDecodeError, DocumentValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc

    result = _ingest(f"raw/uploads/{parsed.id}.md", content)
    document = {
        "id": parsed.id,
        "title": parsed.title,
        "version": parsed.version,
        "system": parsed.system,
        "environment": parsed.environment,
    }
    return result, document


def _health() -> tuple[str, str]:
    endpoints = (
        "http://127.0.0.1:4566/",
        "http://127.0.0.1:8080/health",
        "http://127.0.0.1:8081/health",
    )
    ready = 0
    for url in endpoints:
        try:
            response = httpx.get(url, timeout=1.0)
            ready += int(response.status_code < 500)
        except httpx.HTTPError:
            pass
    if ready == len(endpoints):
        return "Local ready", "ready"
    return f"Degraded {ready}/{len(endpoints)}", "degraded"


def _documents() -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        try:
            metadata, _ = parse_document(path.read_text(encoding="utf-8"))
        except DocumentValidationError:
            continue
        documents.append(
            {
                "id": metadata.id,
                "title": metadata.title,
                "version": metadata.version,
                "system": metadata.system,
                "environment": metadata.environment,
            }
        )
    by_id = {item["id"]: item for item in documents}
    for item in SESSION_UPLOADS:
        by_id[item["id"]] = item
    return list(by_id.values())


def _render_documents(documents: list[dict[str, str]]) -> str:
    return "".join(
        "<div class='document'>"
        f"<strong>{html.escape(item['title'])}</strong>"
        f"<div class='document-meta'>v{html.escape(item['version'])} · "
        f"{html.escape(item['system'])} · {html.escape(item['environment'])}</div>"
        "</div>"
        for item in documents
    )


def _render_result(result: dict[str, object] | None, elapsed_ms: int | None) -> str:
    if not result:
        return ""
    status = str(result["status"])
    label = STATUS_LABELS.get(status, status)
    answer = str(result.get("answer") or result.get("clarification_question") or "")
    if not answer and status == "insufficient_evidence":
        answer = (
            "La documentación disponible no aporta evidencia suficiente "
            "para responder con seguridad."
        )
    if not answer and status == "safety_blocked":
        answer = "La consulta fue bloqueada por las reglas de seguridad del sistema."
    citations = "".join(
        "<div class='citation'>"
        f"{html.escape(str(item['title']))} · v{html.escape(str(item['version']))}"
        "</div>"
        for item in result.get("citations", [])
    )
    timing = f"<span class='metric'>{elapsed_ms} ms</span>" if elapsed_ms is not None else ""
    return (
        "<article class='message assistant-message'>"
        "<div class='message-meta'>RAG Ops Guard</div>"
        "<div class='message-body'>"
        f"<div class='response-top'><span class='badge {status}'>"
        f"{html.escape(label)}</span>{timing}</div>"
        f"{html.escape(answer)}"
        f"<div class='citations'>{citations}</div>"
        "</div></article>"
    )


def _page(
    result: dict[str, object] | None = None,
    notice: str = "",
    question: str = "",
    system: str = "",
    environment: str = "production",
    elapsed_ms: int | None = None,
) -> str:
    documents = _documents()
    health, health_class = _health()
    notice_html = (
        f"<div class='notice success' role='status'>{html.escape(notice)}</div>" if notice else ""
    )
    replacements = {
        "{{HEALTH}}": html.escape(health),
        "{{HEALTH_CLASS}}": health_class,
        "{{NOTICE}}": notice_html,
        "{{DOCUMENT_COUNT}}": str(len(documents)),
        "{{DOCUMENTS}}": _render_documents(documents),
        "{{RESULT}}": _render_result(result, elapsed_ms),
        "{{QUESTION}}": html.escape(question),
        "{{SYSTEM}}": html.escape(system),
        "{{ENV_ANY}}": "selected" if not environment else "",
        "{{ENV_PRODUCTION}}": "selected" if environment == "production" else "",
        "{{ENV_STAGING}}": "selected" if environment == "staging" else "",
    }
    page = TEMPLATE
    for source, target in replacements.items():
        page = page.replace(source, target)
    return page


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.post("/query", response_class=HTMLResponse)
def query(
    question: Annotated[str, Form(min_length=3, max_length=2000)],
    system: Annotated[str, Form()] = "",
    environment: Annotated[str, Form()] = "",
) -> str:
    env = environment if environment in {"production", "staging"} else None
    started = time.perf_counter()
    response = query_workflow().invoke(
        QueryRequest(
            question=question,
            context=QueryContext(system=system or None, environment=env),
        )
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _page(
        response.model_dump(mode="json"),
        question=question,
        system=system,
        environment=environment,
        elapsed_ms=elapsed_ms,
    )


@app.post("/documents", response_class=HTMLResponse)
def upload(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()] = "",
    system: Annotated[str, Form()] = "general",
    environment: Annotated[str, Form()] = "all",
    document_type: Annotated[str, Form()] = "architecture",
    version: Annotated[str, Form()] = "1.0",
    authority: Annotated[int, Form()] = 50,
) -> str:
    started = time.perf_counter()
    result, document = _prepare_upload(
        file, title, system, environment, document_type, version, authority
    )
    SESSION_UPLOADS.append(document)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    notice = (
        f"{document['title']} listo: {result['chunks']} chunks ingeridos y vectorizados "
        f"en {elapsed_ms} ms."
    )
    return _page(notice=notice)


@app.post("/demo/load", response_class=HTMLResponse)
def load_demo() -> str:
    started = time.perf_counter()
    count = 0
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        relative = path.relative_to(KNOWLEDGE_DIR).as_posix()
        _ingest(f"raw/{relative}", path.read_text(encoding="utf-8"))
        count += 1
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _page(notice=f"Corpus demo listo: {count} documentos en {elapsed_ms} ms.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
