from __future__ import annotations

from io import BytesIO

from botocore.exceptions import ClientError

from rag_ops_guard.configstore.secrets_manager_backend import (
    SecretsManagerResolver,
    SecretsManagerSecretBackend,
    secret_name,
)


class FakeSecretsManager:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.get_calls = 0

    def put_secret_value(self, *, SecretId: str, SecretBinary: bytes) -> None:
        if SecretId not in self.values:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "PutSecretValue",
            )
        self.values[SecretId] = bytes(SecretBinary)

    def create_secret(self, *, Name: str, SecretBinary: bytes, Description: str) -> None:
        del Description
        self.values[Name] = bytes(SecretBinary)

    def delete_secret(self, *, SecretId: str, ForceDeleteWithoutRecovery: bool) -> None:
        assert ForceDeleteWithoutRecovery is True
        if SecretId not in self.values:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "DeleteSecret",
            )
        del self.values[SecretId]

    def get_secret_value(self, *, SecretId: str) -> dict[str, object]:
        self.get_calls += 1
        return {"SecretBinary": self.values[SecretId]}


def test_recoverable_secret_backend_uses_canonical_secrets_manager_names() -> None:
    client = FakeSecretsManager()
    backend = SecretsManagerSecretBackend(client=client)

    backend.set_secret(scope="TENANT#tenant-a", key_name="LangSmith", secret=b"secret")

    ref = secret_name("TENANT#tenant-a", "LangSmith")
    assert ref == "rag-ops-guard/tenant/tenant-a/langsmith"
    assert client.values == {ref: b"secret"}
    backend.delete_secret(scope="TENANT#tenant-a", key_name="LangSmith")
    assert client.values == {}


def test_runtime_secret_resolver_caches_plaintext_only_in_memory() -> None:
    client = FakeSecretsManager()
    client.values["rag-ops-guard/runtime/langsmith-api-key"] = b"abc"
    resolver = SecretsManagerResolver(client=client, ttl_seconds=60)

    assert resolver.get_text("rag-ops-guard/runtime/langsmith-api-key") == "abc"
    assert resolver.get_text("rag-ops-guard/runtime/langsmith-api-key") == "abc"
    assert client.get_calls == 1
