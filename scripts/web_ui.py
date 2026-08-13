from __future__ import annotations

import argparse
import html
import re
from datetime import date
from pathlib import Path
from typing import Annotated

import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from rag_ops_guard.app import ingestion_service, object_store, query_workflow
from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.ingestion.metadata import MAX_DOCUMENT_BYTES, parse_document

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge-base"
app = FastAPI(title="RAG Ops Guard", docs_url=None, redoc_url=None)


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
) -> dict[str, object]:
    filename = Path(file.filename or "document.md").name
    if Path(filename).suffix.lower() not in {".md", ".markdown"}:
        raise HTTPException(415, "Only Markdown files are supported in beta 1")
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
        metadata, _ = parse_document(content)
    except (UnicodeDecodeError, DocumentValidationError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return _ingest(f"raw/uploads/{metadata.id}.md", content)


def _page(result: dict[str, object] | None = None, notice: str = "") -> str:
    answer = ""
    if result:
        status = html.escape(str(result["status"]))
        text = html.escape(str(result.get("answer") or result.get("clarification_question") or ""))
        citations = "".join(
            f"<li>{html.escape(str(c['title']))} · v{html.escape(str(c['version']))}</li>"
            for c in result.get("citations", [])
        )
        answer = f"<section class='answer'><b>{status}</b><p>{text}</p><ul>{citations}</ul></section>"
    message = f"<p class='notice'>{html.escape(notice)}</p>" if notice else ""
    return f"""
<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>RAG Ops Guard</title><style>
body{{margin:0;font-family:system-ui;background:#f3f0e8;color:#151515}}main{{max-width:1100px;margin:auto;padding:28px}}.grid{{display:grid;grid-template-columns:280px 1fr;gap:24px}}aside,section,form{{background:#fffdf7;border:1px solid #222;padding:18px}}h1{{margin-top:0}}label{{display:grid;gap:5px;margin:9px 0;font-size:13px;font-weight:650}}input,select,textarea,button{{font:inherit;padding:9px;border:1px solid #222;background:#fffdf7;color:#151515;width:100%;box-sizing:border-box}}button{{background:#151515;color:#fff;cursor:pointer}}.query{{display:grid;grid-template-columns:1fr 150px;gap:10px}}.answer{{margin-bottom:16px}}.notice{{padding:10px;border:1px solid #222;background:#fffdf7}}small{{color:#666}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}.query{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>RAG Ops Guard</h1><p>FastAPI → LangGraph → Qwen → Floci</p>{message}<div class='grid'><aside><h2>Knowledge</h2><small>Demo corpus is loaded automatically by make ui.</small><form action='/documents' method='post' enctype='multipart/form-data'><h3>Add Markdown</h3><input name='file' type='file' accept='.md,.markdown' required><label>Title<input name='title'></label><label>System<input name='system' value='general'></label><label>Environment<select name='environment'><option value='all'>All</option><option value='production'>Production</option><option value='staging'>Staging</option></select></label><label>Type<select name='document_type'><option>runbook</option><option>api</option><option>sla</option><option>incident</option><option>postmortem</option><option>architecture</option></select></label><label>Version<input name='version' value='1.0'></label><label>Authority<input name='authority' type='number' min='0' max='100' value='50'></label><button>Ingest</button></form><form action='/demo/load' method='post'><button>Reload demo corpus</button></form></aside><section>{answer}<form action='/query' method='post'><h2>Ask</h2><textarea name='question' rows='4' maxlength='2000' required placeholder='Ask about retries, incidents, SLAs, APIs...'></textarea><div class='query'><input name='system' value='payments' placeholder='system'><select name='environment'><option value=''>Any</option><option value='production' selected>Production</option><option value='staging'>Staging</option></select></div><button>Ask RAG Ops Guard</button></form></section></div></main></body></html>
"""


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
    response = query_workflow().invoke(
        QueryRequest(question=question, context=QueryContext(system=system or None, environment=env))
    )
    return _page(response.model_dump(mode="json"))


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
    result = _prepare_upload(file, title, system, environment, document_type, version, authority)
    return _page(notice=f"{result['document_id']} {result['status']} ({result['chunks']} chunks)")


@app.post("/demo/load", response_class=HTMLResponse)
def load_demo() -> str:
    count = 0
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        relative = path.relative_to(KNOWLEDGE_DIR).as_posix()
        _ingest(f"raw/{relative}", path.read_text(encoding="utf-8"))
        count += 1
    return _page(notice=f"{count} demo documents ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
