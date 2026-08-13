from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import boto3
import httpx
import uvicorn
import yaml
from botocore.config import Config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_ops_guard.app import ingestion_service, object_store, query_workflow
from rag_ops_guard.config import get_settings
from rag_ops_guard.domain.errors import DocumentValidationError
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.ingestion.metadata import MAX_DOCUMENT_BYTES, parse_document

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
KNOWLEDGE_DIR = ROOT / "knowledge-base"
ALLOWED_ENVIRONMENTS = {"production", "staging", "all"}
ALLOWED_TYPES = {"runbook", "api", "sla", "incident", "postmortem", "architecture"}

app = FastAPI(title="RAG Ops Guard Web Console", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class UiQuery(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    system: str | None = None
    environment: Literal["production", "staging"] | None = None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=Config(s3={"addressing_style": "path"}),
    )


def _with_frontmatter(
    content: str,
    filename: str,
    title: str,
    system: str,
    environment: str,
    document_type: str,
    version: str,
    authority: int,
) -> str:
    if content.lstrip().startswith("---"):
        parse_document(content)
        return content

    if environment not in ALLOWED_ENVIRONMENTS:
        raise DocumentValidationError("invalid environment")
    if document_type not in ALLOWED_TYPES:
        raise DocumentValidationError("invalid document type")
    if not 0 <= authority <= 100:
        raise DocumentValidationError("authority must be between 0 and 100")

    resolved_title = title.strip() or Path(filename).stem.replace("-", " ").title()
    logical_id = _slug(resolved_title)
    metadata = {
        "id": f"{logical_id}-v{_slug(version)}",
        "logical_id": logical_id,
        "title": resolved_title,
        "version": version.strip() or "1.0",
        "status": "active",
        "effective_date": date.today().isoformat(),
        "system": system.strip() or "general",
        "environment": environment,
        "document_type": document_type,
        "authority": authority,
        "supersedes": [],
    }
    header = yaml.safe_dump(metadata, sort_keys=False).strip()
    result = f"---\n{header}\n---\n\n{content.strip()}\n"
    parse_document(result)
    return result


def _ingest(key: str, content: str) -> dict[str, object]:
    object_store().put_text(key, content, "text/markdown")
    response = ingestion_service().ingest(key)
    return response.model_dump(mode="json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    checks = {
        "floci": "http://127.0.0.1:4566/",
        "generation": "http://127.0.0.1:8080/health",
        "embeddings": "http://127.0.0.1:8081/health",
    }
    services: dict[str, str] = {}
    for name, url in checks.items():
        try:
            response = httpx.get(url, timeout=2.0)
            services[name] = "ready" if response.status_code < 500 else "unavailable"
        except httpx.HTTPError:
            services[name] = "unavailable"
    overall = "ready" if all(value == "ready" for value in services.values()) else "degraded"
    return {"status": overall, "services": services}


@app.post("/api/query")
def query(payload: UiQuery) -> dict[str, object]:
    try:
        response = query_workflow().invoke(
            QueryRequest(
                question=payload.question,
                context=QueryContext(
                    system=payload.system or None,
                    environment=payload.environment,
                ),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query failed") from exc
    return response.model_dump(mode="json")


@app.get("/api/documents")
def documents() -> dict[str, object]:
    settings = get_settings()
    response = _s3_client().list_objects_v2(Bucket=settings.s3_document_bucket, Prefix="raw/")
    items: list[dict[str, object]] = []
    for entry in response.get("Contents", []):
        key = str(entry["Key"])
        try:
            metadata, _ = parse_document(object_store().get_text(key))
        except (DocumentValidationError, UnicodeDecodeError):
            continue
        items.append(
            {
                "key": key,
                "logical_id": metadata.logical_id,
                "title": metadata.title,
                "version": metadata.version,
                "status": metadata.status.value,
                "system": metadata.system,
                "environment": metadata.environment,
                "document_type": metadata.document_type.value,
            }
        )
    items.sort(key=lambda item: (str(item["title"]), str(item["version"])), reverse=True)
    return {"count": len(items), "documents": items}


@app.post("/api/documents")
def upload_document(
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()] = "",
    system: Annotated[str, Form()] = "general",
    environment: Annotated[str, Form()] = "all",
    document_type: Annotated[str, Form()] = "architecture",
    version: Annotated[str, Form()] = "1.0",
    authority: Annotated[int, Form()] = 50,
) -> dict[str, object]:
    filename = Path(file.filename or "document.md").name
    if Path(filename).suffix.lower() not in {".md", ".markdown"}:
        raise HTTPException(status_code=415, detail="Only Markdown files are supported in beta 1")

    raw = file.file.read(MAX_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 256 KiB limit")
    try:
        content = raw.decode("utf-8")
        prepared = _with_frontmatter(
            content,
            filename,
            title,
            system,
            environment,
            document_type,
            version,
            authority,
        )
        metadata, _ = parse_document(prepared)
        result = _ingest(f"raw/uploads/{metadata.id}.md", prepared)
    except (UnicodeDecodeError, DocumentValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"document": result}


@app.post("/api/demo/load")
def load_demo_corpus() -> dict[str, object]:
    results: list[dict[str, object]] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        relative = path.relative_to(KNOWLEDGE_DIR).as_posix()
        results.append(_ingest(f"raw/{relative}", path.read_text(encoding="utf-8")))
    return {"count": len(results), "documents": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RAG Ops Guard Web Console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
