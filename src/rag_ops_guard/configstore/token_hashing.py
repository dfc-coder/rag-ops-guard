from __future__ import annotations

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError

ARGON2_MEMORY_COST_KIB = 64 * 1024
ARGON2_TIME_COST = 3
ARGON2_PARALLELISM = 4


class TenantTokenHasher:
    """Hash and verify tenant tokens without logging or retaining cleartext values."""

    def __init__(self) -> None:
        self._hasher = PasswordHasher(
            time_cost=ARGON2_TIME_COST,
            memory_cost=ARGON2_MEMORY_COST_KIB,
            parallelism=ARGON2_PARALLELISM,
            hash_len=32,
            salt_len=16,
            type=Type.ID,
        )

    def hash_token(self, token: str) -> str:
        if not token:
            raise ValueError("tenant token must not be empty")
        return self._hasher.hash(token)

    def verify_token(self, *, encoded_hash: str, token: str) -> bool:
        if not encoded_hash or not token:
            return False
        try:
            return bool(self._hasher.verify(encoded_hash, token))
        except (VerificationError, InvalidHashError):
            return False
