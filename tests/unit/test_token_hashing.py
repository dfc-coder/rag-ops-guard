from __future__ import annotations

import logging

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher


def test_tenant_token_hash_uses_required_argon2id_parameters() -> None:
    hasher = TenantTokenHasher()

    encoded = hasher.hash_token("tenant-token-value")

    assert encoded.startswith("$argon2id$")
    assert "m=65536,t=3,p=4" in encoded
    assert "tenant-token-value" not in encoded


def test_tenant_token_verification_accepts_only_matching_token() -> None:
    hasher = TenantTokenHasher()
    encoded = hasher.hash_token("tenant-token-value")

    assert hasher.verify_token(encoded_hash=encoded, token="tenant-token-value") is True
    assert hasher.verify_token(encoded_hash=encoded, token="wrong-token") is False
    assert hasher.verify_token(encoded_hash="not-an-argon2-hash", token="tenant-token-value") is False


def test_tenant_token_operations_do_not_log_token(caplog: object) -> None:
    logger = logging.getLogger()
    before_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        hasher = TenantTokenHasher()
        encoded = hasher.hash_token("tenant-token-value")
        hasher.verify_token(encoded_hash=encoded, token="tenant-token-value")
    finally:
        logger.setLevel(before_level)

    records = getattr(caplog, "records")
    assert all("tenant-token-value" not in record.getMessage() for record in records)
