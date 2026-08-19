from __future__ import annotations

from dataclasses import replace

import pytest

from rag_ops_guard.configstore.envelope import (
    EnvelopeAuthenticationError,
    decrypt_secret,
    encrypt_secret,
)

DEK = b"k" * 32
SECRET = b"super-secret-value"
SCOPE = "tenant-a"
KEY_NAME = "openai_api_key"
DEK_VERSION = 7


def test_encrypt_decrypt_round_trip_with_same_aad_context() -> None:
    envelope = encrypt_secret(
        SECRET,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
        dek_version=DEK_VERSION,
    )

    plaintext = decrypt_secret(
        envelope,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
    )

    assert plaintext == SECRET


@pytest.mark.parametrize(
    ("scope", "key_name", "dek_version"),
    [
        ("tenant-b", KEY_NAME, DEK_VERSION),
        (SCOPE, "anthropic_api_key", DEK_VERSION),
        (SCOPE, KEY_NAME, DEK_VERSION + 1),
    ],
)
def test_decrypt_rejects_aad_context_swap(
    scope: str,
    key_name: str,
    dek_version: int,
) -> None:
    envelope = encrypt_secret(
        SECRET,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
        dek_version=DEK_VERSION,
    )
    candidate = replace(envelope, dek_version=dek_version)

    with pytest.raises(EnvelopeAuthenticationError):
        decrypt_secret(
            candidate,
            dek=DEK,
            scope=scope,
            key_name=key_name,
        )


def test_encrypt_uses_unique_96_bit_nonces() -> None:
    first = encrypt_secret(
        SECRET,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
        dek_version=DEK_VERSION,
    )
    second = encrypt_secret(
        SECRET,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
        dek_version=DEK_VERSION,
    )

    assert len(first.nonce) == 12
    assert len(second.nonce) == 12
    assert first.nonce != second.nonce


def test_decrypt_rejects_ciphertext_tampering() -> None:
    envelope = encrypt_secret(
        SECRET,
        dek=DEK,
        scope=SCOPE,
        key_name=KEY_NAME,
        dek_version=DEK_VERSION,
    )
    tampered = replace(
        envelope,
        ciphertext=envelope.ciphertext[:-1] + bytes([envelope.ciphertext[-1] ^ 1]),
    )

    with pytest.raises(EnvelopeAuthenticationError):
        decrypt_secret(
            tampered,
            dek=DEK,
            scope=SCOPE,
            key_name=KEY_NAME,
        )


def test_encrypt_rejects_non_256_bit_dek() -> None:
    with pytest.raises(ValueError, match="32-byte DEK"):
        encrypt_secret(
            SECRET,
            dek=b"short",
            scope=SCOPE,
            key_name=KEY_NAME,
            dek_version=DEK_VERSION,
        )
