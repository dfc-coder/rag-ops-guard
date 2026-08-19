from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from rag_ops_guard.tenancy import KeyLayout

APPLICATION = "rag-ops-guard"
PURPOSE = "local-tenant-api-key"


class LocalCredentialError(RuntimeError):
    """Local client credential storage or retrieval failed."""


@dataclass(frozen=True)
class TenantApiCredential:
    key_id: str
    secret: str

    @classmethod
    def parse(cls, api_key: str) -> TenantApiCredential:
        value = api_key.strip()
        key_id, separator, secret = value.partition(".")
        if not separator or not key_id or not secret:
            raise ValueError("tenant API credential must use key_id.secret format")
        return cls(key_id=key_id, secret=secret)

    @property
    def api_key(self) -> str:
        return f"{self.key_id}.{self.secret}"


class SecretServiceCredentialStore:
    """Store local client API keys in the desktop Secret Service via `secret-tool`.

    The cleartext credential is passed to `secret-tool store` over stdin, never as a
    command-line argument, environment variable, repository file, or DynamoDB value.
    """

    def __init__(self, *, binary: str | None = None) -> None:
        resolved = binary or shutil.which("secret-tool")
        if not resolved:
            raise LocalCredentialError(
                "secret-tool is required for local tenant credentials; install the system "
                "Secret Service client (Fedora: libsecret) and retry"
            )
        self._binary = resolved

    @staticmethod
    def _tenant_id(tenant_id: str) -> str:
        return KeyLayout(tenant_id).tenant_id

    def _attributes(self, tenant_id: str) -> list[str]:
        return [
            "application",
            APPLICATION,
            "purpose",
            PURPOSE,
            "tenant",
            self._tenant_id(tenant_id),
        ]

    @staticmethod
    def _error(action: str, result: subprocess.CompletedProcess[str]) -> LocalCredentialError:
        detail = result.stderr.strip() or f"secret-tool exited with status {result.returncode}"
        return LocalCredentialError(f"Secret Service {action} failed: {detail}")

    def get_api_key(self, tenant_id: str) -> str | None:
        result = subprocess.run(
            [self._binary, "lookup", *self._attributes(tenant_id)],
            input=None,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
            return None
        if result.returncode != 0:
            raise self._error("lookup", result)
        value = result.stdout.strip()
        if not value:
            return None
        try:
            return TenantApiCredential.parse(value).api_key
        except ValueError as exc:
            raise LocalCredentialError(
                f"stored tenant credential for {self._tenant_id(tenant_id)!r} is invalid"
            ) from exc

    def require_api_key(self, tenant_id: str) -> str:
        value = self.get_api_key(tenant_id)
        if value is None:
            normalized = self._tenant_id(tenant_id)
            raise LocalCredentialError(
                f"no local credential exists for tenant {normalized!r}; "
                f"run `make tenant-create TENANT={normalized}` first"
            )
        return value

    def set_api_key(self, tenant_id: str, api_key: str) -> None:
        normalized = self._tenant_id(tenant_id)
        value = TenantApiCredential.parse(api_key).api_key
        result = subprocess.run(
            [
                self._binary,
                "store",
                f"--label=RAG Ops Guard tenant {normalized}",
                *self._attributes(normalized),
            ],
            input=f"{value}\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise self._error("store", result)

    def delete_api_key(self, tenant_id: str) -> None:
        result = subprocess.run(
            [self._binary, "clear", *self._attributes(tenant_id)],
            input=None,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 1 and not result.stderr.strip():
            return
        if result.returncode != 0:
            raise self._error("clear", result)


def require_local_api_key(tenant_id: str = "default") -> str:
    """Resolve one local client credential exclusively from the OS Secret Service."""

    return SecretServiceCredentialStore().require_api_key(tenant_id)
