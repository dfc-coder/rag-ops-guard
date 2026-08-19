from __future__ import annotations

import argparse
import os
from typing import Any

import boto3
from botocore.config import Config

from rag_ops_guard.tenancy import KeyLayout

LEGACY_PREFIXES = ("raw/", "chunks/", "manifests/")


def target_key(layout: KeyLayout, legacy_key: str) -> str:
    if not any(legacy_key.startswith(prefix) for prefix in LEGACY_PREFIXES):
        raise ValueError(f"unsupported legacy key: {legacy_key}")
    return f"{layout.tenant_prefix}{legacy_key}"


def _client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        config=Config(s3={"addressing_style": "path"}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy legacy single-tenant S3 keys into Phase 4 layout")
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--apply", action="store_true", help="perform copies; default is dry-run")
    args = parser.parse_args()

    layout = KeyLayout(args.tenant_id)
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    s3 = _client()
    copies: list[tuple[str, str]] = []
    for prefix in LEGACY_PREFIXES:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for item in response.get("Contents", []):
            source = str(item["Key"])
            copies.append((source, target_key(layout, source)))

    for source, target in copies:
        print(f"{source} -> {target}")
        if args.apply:
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": source}, Key=target)
    print(f"migration {'applied' if args.apply else 'dry-run'}: {len(copies)} objects")


if __name__ == "__main__":
    main()
