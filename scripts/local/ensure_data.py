from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from rag_ops_guard.domain.models import Manifest
from rag_ops_guard.ingestion.manifest import document_sha256, manifest_key
from rag_ops_guard.ingestion.metadata import parse_document
from rag_ops_guard.tenancy import KeyLayout

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
DOC_BUCKET = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
VECTOR_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "rag-ops-guard-vectors-local")
VECTOR_DIMENSION = int(os.environ.get("VECTOR_DIMENSION", "1024"))
LAYOUT = KeyLayout(os.environ.get("RAG_OPS_TENANT_ID", "default"))
VECTOR_INDEX = LAYOUT.vector_index(os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-v1"))


@dataclass(frozen=True)
class LocalDocument:
    manifest_key: str
    digest: str


def client(service: str, **kwargs: object) -> Any:
    return boto3.client(
        service,
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        **kwargs,
    )


def ensure_document_bucket() -> bool:
    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    try:
        s3.head_bucket(Bucket=DOC_BUCKET)
        return False
    except ClientError:
        s3.create_bucket(Bucket=DOC_BUCKET)
        return True


def ensure_vector_index() -> bool:
    vectors = client("s3vectors")
    created = False

    try:
        vectors.get_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    except ClientError:
        vectors.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        created = True

    try:
        vectors.get_index(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX)
    except ClientError:
        vectors.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=VECTOR_DIMENSION,
            distanceMetric="cosine",
        )
        created = True

    return created


def local_documents() -> list[LocalDocument]:
    documents: list[LocalDocument] = []
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        metadata, _ = parse_document(content)
        documents.append(
            LocalDocument(
                manifest_key=manifest_key(metadata.logical_id, metadata.version, LAYOUT),
                digest=document_sha256(content),
            )
        )
    return documents


def corpus_ready() -> bool:
    """Verify every repository document manifest and every referenced vector for one tenant."""
    documents = local_documents()
    if not documents:
        return False

    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    vectors = client("s3vectors")
    vector_keys: list[str] = []

    for document in documents:
        try:
            response = s3.get_object(Bucket=DOC_BUCKET, Key=document.manifest_key)
        except ClientError:
            return False
        payload = response["Body"].read().decode("utf-8")
        manifest = Manifest.model_validate_json(payload)
        if manifest.content_sha256 != document.digest or not manifest.vector_keys:
            return False
        vector_keys.extend(manifest.vector_keys)

    for offset in range(0, len(vector_keys), 100):
        batch = vector_keys[offset : offset + 100]
        response = vectors.get_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            keys=batch,
        )
        returned = {str(item["key"]) for item in response.get("vectors", [])}
        if not set(batch).issubset(returned):
            return False

    return True


def clear_local_manifests() -> None:
    """Force tenant ingestion to regenerate vectors when the selected vector index is stale."""
    s3 = client("s3", config=Config(s3={"addressing_style": "path"}))
    for document in local_documents():
        with contextlib.suppress(ClientError):
            s3.delete_object(Bucket=DOC_BUCKET, Key=document.manifest_key)


def rebuild_demo_data() -> None:
    print(f"tenant {LAYOUT.tenant_id} knowledge base incomplete or stale; rebuilding demo corpus")
    clear_local_manifests()
    subprocess.run([sys.executable, "scripts/seed.py"], check=True)
    subprocess.run([sys.executable, "scripts/demo_prepare.py"], check=True)
    if not corpus_ready():
        raise SystemExit("local knowledge base rebuild completed but integrity verification failed")


def main() -> None:
    document_bucket_created = ensure_document_bucket()
    vector_index_created = ensure_vector_index()

    if document_bucket_created or vector_index_created or not corpus_ready():
        rebuild_demo_data()
        return

    print(f"tenant {LAYOUT.tenant_id} knowledge base: ready (all manifests and vectors verified)")


if __name__ == "__main__":
    main()
