from __future__ import annotations

from typing import Any

import pytest

from rag_ops_guard.configstore.key_provider import KmsKeyProvider, LocalKeyProvider


class FakeKmsClient:
    def __init__(self) -> None:
        self.generated_for: str | None = None
        self.decrypted_for: str | None = None
        self.decrypted_blob: bytes | None = None

    def generate_data_key(self, **kwargs: Any) -> dict[str, object]:
        self.generated_for = str(kwargs["KeyId"])
        assert kwargs["KeySpec"] == "AES_256"
        return {
            "Plaintext": b"p" * 32,
            "CiphertextBlob": b"wrapped-dek",
        }

    def decrypt(self, **kwargs: Any) -> dict[str, object]:
        self.decrypted_for = str(kwargs["KeyId"])
        self.decrypted_blob = bytes(kwargs["CiphertextBlob"])
        return {"Plaintext": b"p" * 32}


def test_kms_key_provider_generates_32_byte_dek_and_unwraps_it() -> None:
    client = FakeKmsClient()
    provider = KmsKeyProvider(client=client)

    generated = provider.generate_data_key(kek_ref="alias/rag-ops-config")

    assert generated.plaintext == b"p" * 32
    assert generated.wrapped == b"wrapped-dek"
    assert client.generated_for == "alias/rag-ops-config"

    plaintext = provider.unwrap(
        kek_ref="alias/rag-ops-config",
        wrapped_key=generated.wrapped,
    )

    assert plaintext == generated.plaintext
    assert client.decrypted_for == "alias/rag-ops-config"
    assert client.decrypted_blob == generated.wrapped


def test_kms_key_provider_rejects_non_256_bit_plaintext() -> None:
    class ShortKeyClient(FakeKmsClient):
        def generate_data_key(self, **kwargs: Any) -> dict[str, object]:
            return {"Plaintext": b"short", "CiphertextBlob": b"wrapped"}

    provider = KmsKeyProvider(client=ShortKeyClient())

    with pytest.raises(ValueError, match="32 bytes"):
        provider.generate_data_key(kek_ref="key")


@pytest.mark.parametrize("app_env", ["aws", "ci", "local-observed"])
def test_local_key_provider_refuses_non_local_environment(app_env: str) -> None:
    with pytest.raises(RuntimeError, match="app_env=local"):
        LocalKeyProvider(app_env=app_env)
