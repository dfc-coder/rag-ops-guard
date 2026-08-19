from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.config import Config

from rag_ops_guard.tenancy import KeyLayout


def main() -> None:
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    layout = KeyLayout(os.environ.get("RAG_OPS_TENANT_ID", "default"))
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(s3={"addressing_style": "path"}),
    )
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        key = layout.raw_key(path.relative_to("knowledge-base").as_posix())
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=path.read_bytes(),
            ContentType="text/markdown",
        )
        print(key)


if __name__ == "__main__":
    main()
