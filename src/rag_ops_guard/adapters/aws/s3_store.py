from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class S3ObjectStore:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        region_name: str,
        access_key: str,
        secret_key: str,
    ) -> None:
        self._bucket = bucket
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(s3={"addressing_style": "path"}),
        )

    def get_text(self, key: str) -> str:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read().decode("utf-8")

    def put_text(self, key: str, content: str, content_type: str = "text/plain") -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
