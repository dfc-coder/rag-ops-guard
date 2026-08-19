from __future__ import annotations

from typing import cast

from rag_ops_guard.configstore.envelope import SecretEnvelope, decrypt_secret, encrypt_secret
from rag_ops_guard.configstore.key_provider import KeyProvider, KeyRewrapper
from rag_ops_guard.configstore.secret_store import (
    DekRecord,
    EncryptedSecretRecord,
    SecretRecordStore,
)


class EnvelopeSecretService:
    """Coordinate envelope encryption without persisting plaintext key material."""

    def __init__(
        self,
        *,
        store: SecretRecordStore,
        key_provider: KeyProvider,
        key_rewrapper: KeyRewrapper | None = None,
    ) -> None:
        self._store = store
        self._key_provider = key_provider
        if key_rewrapper is None and callable(getattr(key_provider, "rewrap", None)):
            key_rewrapper = cast(KeyRewrapper, key_provider)
        self._key_rewrapper = key_rewrapper

    def _latest_dek(self, *, scope: str, initial_kek_ref: str) -> tuple[DekRecord, bytes]:
        record = self._store.get_latest_dek(scope=scope)
        if record is not None:
            plaintext = self._key_provider.unwrap(
                kek_ref=record.kek_ref,
                wrapped_key=record.wrapped_dek,
            )
            return record, plaintext

        generated = self._key_provider.generate_data_key(kek_ref=initial_kek_ref)
        record = DekRecord(
            scope=scope,
            version=1,
            kek_ref=initial_kek_ref,
            wrapped_dek=generated.wrapped,
        )
        self._store.create_dek(record)
        return record, generated.plaintext

    def set_secret(
        self,
        *,
        scope: str,
        key_name: str,
        secret: bytes,
        kek_ref: str,
    ) -> None:
        dek_record, plaintext_dek = self._latest_dek(
            scope=scope,
            initial_kek_ref=kek_ref,
        )
        envelope = encrypt_secret(
            secret,
            dek=plaintext_dek,
            scope=scope,
            key_name=key_name,
            dek_version=dek_record.version,
        )
        self._store.put_secret(
            EncryptedSecretRecord(
                scope=scope,
                key_name=key_name,
                ciphertext=envelope.ciphertext,
                nonce=envelope.nonce,
                dek_version=envelope.dek_version,
            )
        )

    def get_secret(self, *, scope: str, key_name: str) -> bytes | None:
        secret_record = self._store.get_secret(scope=scope, key_name=key_name)
        if secret_record is None:
            return None
        dek_record = self._store.get_dek(scope=scope, version=secret_record.dek_version)
        if dek_record is None:
            raise RuntimeError(
                f"missing DEK version {secret_record.dek_version} for scope {scope!r}"
            )
        plaintext_dek = self._key_provider.unwrap(
            kek_ref=dek_record.kek_ref,
            wrapped_key=dek_record.wrapped_dek,
        )
        return decrypt_secret(
            SecretEnvelope(
                ciphertext=secret_record.ciphertext,
                nonce=secret_record.nonce,
                dek_version=secret_record.dek_version,
            ),
            dek=plaintext_dek,
            scope=scope,
            key_name=key_name,
        )

    def rotate_kek(self, *, scope: str, new_kek_ref: str) -> int:
        if self._key_rewrapper is None:
            raise RuntimeError("configured key provider does not support KEK re-wrapping")
        rotated = 0
        for record in self._store.list_deks(scope=scope):
            if record.kek_ref == new_kek_ref:
                continue
            wrapped = self._key_rewrapper.rewrap(
                source_kek_ref=record.kek_ref,
                target_kek_ref=new_kek_ref,
                wrapped_key=record.wrapped_dek,
            )
            self._store.replace_dek(
                DekRecord(
                    scope=record.scope,
                    version=record.version,
                    kek_ref=new_kek_ref,
                    wrapped_dek=wrapped,
                )
            )
            rotated += 1
        return rotated

    def rotate_dek(self, *, scope: str, kek_ref: str) -> int:
        latest = self._store.get_latest_dek(scope=scope)
        if latest is None:
            raise ValueError(f"cannot rotate DEK for scope {scope!r}: no DEK exists")

        new_version = latest.version + 1
        generated = self._key_provider.generate_data_key(kek_ref=kek_ref)
        self._store.create_dek(
            DekRecord(
                scope=scope,
                version=new_version,
                kek_ref=kek_ref,
                wrapped_dek=generated.wrapped,
            )
        )

        plaintext_deks: dict[int, bytes] = {}
        for secret_record in self._store.list_secrets(scope=scope):
            old_dek = plaintext_deks.get(secret_record.dek_version)
            if old_dek is None:
                old_dek_record = self._store.get_dek(
                    scope=scope,
                    version=secret_record.dek_version,
                )
                if old_dek_record is None:
                    raise RuntimeError(
                        f"missing DEK version {secret_record.dek_version} for scope {scope!r}"
                    )
                old_dek = self._key_provider.unwrap(
                    kek_ref=old_dek_record.kek_ref,
                    wrapped_key=old_dek_record.wrapped_dek,
                )
                plaintext_deks[secret_record.dek_version] = old_dek

            plaintext = decrypt_secret(
                SecretEnvelope(
                    ciphertext=secret_record.ciphertext,
                    nonce=secret_record.nonce,
                    dek_version=secret_record.dek_version,
                ),
                dek=old_dek,
                scope=scope,
                key_name=secret_record.key_name,
            )
            new_envelope = encrypt_secret(
                plaintext,
                dek=generated.plaintext,
                scope=scope,
                key_name=secret_record.key_name,
                dek_version=new_version,
            )
            self._store.put_secret(
                EncryptedSecretRecord(
                    scope=scope,
                    key_name=secret_record.key_name,
                    ciphertext=new_envelope.ciphertext,
                    nonce=new_envelope.nonce,
                    dek_version=new_version,
                )
            )

        return new_version
