from __future__ import annotations

from rag_ops_guard.configstore.secret_service import EnvelopeSecretService


class EnvelopeSecretBackend:
    """Phase-3 compatibility adapter behind the Phase-5 SecretBackend port."""

    def __init__(self, *, service: EnvelopeSecretService, kek_ref: str) -> None:
        if not kek_ref.strip():
            raise ValueError("kek_ref must not be empty")
        self._service = service
        self._kek_ref = kek_ref

    def set_secret(self, *, scope: str, key_name: str, secret: bytes) -> None:
        self._service.set_secret(
            scope=scope,
            key_name=key_name,
            secret=secret,
            kek_ref=self._kek_ref,
        )

    def delete_secret(self, *, scope: str, key_name: str) -> None:
        self._service.delete_secret(scope=scope, key_name=key_name)
