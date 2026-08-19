from __future__ import annotations

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class S3ObjectStore:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._bucket = bucket
        client_kwargs: dict[str, object] = {
            "config": Config(s3={"addressing_style": "path"}),
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if region_name:
            client_kwargs["region_name"] = region_name
        if access_key is not None:
            client_kwargs["aws_access_key_id"] = access_key
        if secret_key is not None:
            client_kwargs["aws_secret_access_key"] = secret_key
        self._client: Any = boto3.client("s3", **client_kwargs)

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
