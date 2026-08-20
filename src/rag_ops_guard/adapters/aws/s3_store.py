from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class S3ObjectStore:
    def __init__(self, bucket: str, *, client: Any | None = None) -> None:
        self._bucket = bucket
        self._client: Any = client or boto3.client(
            "s3",
            config=Config(s3={"addressing_style": "path"}),
        )

    def get_text(self, key: str) -> str:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        payload: bytes = response["Body"].read()
        return payload.decode("utf-8")

    def put_text(self, key: str, content: str, content_type: str = "text/plain") -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType=content_type,
        )

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._client.list_objects_v2(**kwargs)
            keys.extend(str(item["Key"]) for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            token = str(response["NextContinuationToken"])
        return keys
