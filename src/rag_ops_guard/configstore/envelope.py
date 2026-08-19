from __future__ import annotations

import secrets
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEK_SIZE_BYTES = 32
NONCE_SIZE_BYTES = 12


@dataclass(frozen=True)
class SecretEnvelope:
    ciphertext: bytes
    nonce: bytes
    dek_version: int


class EnvelopeAuthenticationError(ValueError):
    """Raised when an encrypted secret cannot be authenticated for its context."""


def _require_dek(dek: bytes) -> bytes:
    if len(dek) != DEK_SIZE_BYTES:
        raise ValueError(f"AES-256-GCM requires a 32-byte DEK; got {len(dek)}")
    return dek


def _encode_aad_part(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def _build_aad(*, scope: str, key_name: str, dek_version: int) -> bytes:
    """Build an unambiguous encoding of scope || key_name || dek_version."""
    return b"".join(
        (
            _encode_aad_part(scope.encode("utf-8")),
            _encode_aad_part(key_name.encode("utf-8")),
            _encode_aad_part(str(dek_version).encode("ascii")),
        )
    )


def encrypt_secret(
    secret: bytes,
    *,
    dek: bytes,
    scope: str,
    key_name: str,
    dek_version: int,
) -> SecretEnvelope:
    key = _require_dek(dek)
    nonce = secrets.token_bytes(NONCE_SIZE_BYTES)
    aad = _build_aad(scope=scope, key_name=key_name, dek_version=dek_version)
    ciphertext = AESGCM(key).encrypt(nonce, secret, aad)

    return SecretEnvelope(
        ciphertext=ciphertext,
        nonce=nonce,
        dek_version=dek_version,
    )


def decrypt_secret(
    envelope: SecretEnvelope,
    *,
    dek: bytes,
    scope: str,
    key_name: str,
) -> bytes:
    key = _require_dek(dek)
    if len(envelope.nonce) != NONCE_SIZE_BYTES:
        raise EnvelopeAuthenticationError("secret envelope authentication failed")

    aad = _build_aad(
        scope=scope,
        key_name=key_name,
        dek_version=envelope.dek_version,
    )

    try:
        return AESGCM(key).decrypt(envelope.nonce, envelope.ciphertext, aad)
    except InvalidTag:
        raise EnvelopeAuthenticationError("secret envelope authentication failed") from None
