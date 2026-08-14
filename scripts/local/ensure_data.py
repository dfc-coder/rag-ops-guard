from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
DOC_BUCKET = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
VECTOR_BUCKET = os.environ.get("S3_VECTOR_BUCKET", "rag-ops-guard-vectors-local")
VECTOR_INDEX = os.environ.get("S3_VECTOR_INDEX", "ops-knowledge-v1")
VECTOR_DIMENSION = int(os.environ.get("VECTOR_DIMENSION", "1024"))


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


def has_vectors() -> bool:
    """Probe through QueryVectors because Floci does not currently route ListVectors correctly."""
    vectors = client("s3vectors")
    probe = [0.0] * VECTOR_DIMENSION
    probe[0] = 1.0
    response = vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=VECTOR_INDEX,
        queryVector={"float32": probe},
        topK=1,
    )
    return bool(response.get("vectors"))


def rebuild_demo_data() -> None:
    print("local knowledge base missing; rebuilding demo vectors once")
    subprocess.run([sys.executable, "scripts/seed.py"], check=True)
    subprocess.run([sys.executable, "scripts/demo_prepare.py"], check=True)


def main() -> None:
    document_bucket_created = ensure_document_bucket()
    vector_index_created = ensure_vector_index()

    if document_bucket_created or vector_index_created or not has_vectors():
        rebuild_demo_data()
        return

    print("local knowledge base: ready")


if __name__ == "__main__":
    main()
