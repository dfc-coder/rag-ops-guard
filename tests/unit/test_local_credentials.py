from __future__ import annotations

import subprocess
from collections.abc import Callable

import pytest

from rag_ops_guard.local_credentials import (
    LocalCredentialError,
    SecretServiceCredentialStore,
    TenantApiCredential,
)


def test_tenant_api_credential_round_trip_and_validation() -> None:
    credential = TenantApiCredential.parse("default.secret-value")

    assert credential.key_id == "default"
    assert credential.secret == "secret-value"
    assert credential.api_key == "default.secret-value"

    for invalid in ("", "missing-separator", ".secret", "key."):
        with pytest.raises(ValueError, match="key_id.secret"):
            TenantApiCredential.parse(invalid)


def test_secret_service_store_passes_secret_only_over_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(
        command: list[str],
        *,
        input: str | None = None,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert text is True
        assert capture_output is True
        assert check is False
        calls.append((command, input))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("rag_ops_guard.local_credentials.subprocess.run", fake_run)
    store = SecretServiceCredentialStore(binary="/usr/bin/secret-tool")

    store.set_api_key("tenant-a", "key-a.super-secret")

    command, stdin = calls[0]
    assert command[:3] == ["/usr/bin/secret-tool", "store", "--label=RAG Ops Guard tenant tenant-a"]
    assert "super-secret" not in command
    assert stdin == "key-a.super-secret\n"
    assert command[-2:] == ["tenant", "tenant-a"]


def test_secret_service_store_lookup_and_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="key-a.super-secret\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 1, stdout="", stderr=""),
        )
    )

    def fake_run(
        command: list[str],
        *,
        input: str | None = None,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del command, input, text, capture_output, check
        return next(responses)

    monkeypatch.setattr("rag_ops_guard.local_credentials.subprocess.run", fake_run)
    store = SecretServiceCredentialStore(binary="secret-tool")

    assert store.get_api_key("tenant-a") == "key-a.super-secret"
    store.delete_api_key("tenant-a")
    assert store.get_api_key("tenant-a") is None


def test_missing_secret_service_backend_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver: Callable[[str], str | None] = lambda _name: None
    monkeypatch.setattr("rag_ops_guard.local_credentials.shutil.which", resolver)

    with pytest.raises(LocalCredentialError, match="secret-tool"):
        SecretServiceCredentialStore()


def test_require_api_key_never_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        *,
        input: str | None = None,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del input, text, capture_output, check
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr("rag_ops_guard.local_credentials.subprocess.run", fake_run)
    store = SecretServiceCredentialStore(binary="secret-tool")

    with pytest.raises(LocalCredentialError, match="tenant-create"):
        store.require_api_key("tenant-a")
