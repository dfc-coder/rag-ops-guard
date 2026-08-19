from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_infrastructure_provisioning_does_not_bootstrap_tenant_credentials() -> None:
    provision = _text("scripts/local/provision.py")

    assert "bootstrap_local_tenant" not in provision
    assert "RAG_OPS_API_KEY" not in provision
    assert "TenantTokenHasher" not in provision


def test_local_api_clients_read_tenant_credentials_from_os_secret_store() -> None:
    for path in (
        "scripts/local/connectivity.py",
        "scripts/ingest_corpus.py",
        "scripts/demo.py",
        "evaluation/runners/run_golden.py",
    ):
        source = _text(path)
        assert "RAG_OPS_API_KEY" not in source, path
        assert "require_local_api_key" in source, path


def test_make_up_is_tenant_independent_and_onboarding_is_explicit() -> None:
    makefile = _text("Makefile")

    up_block = makefile[makefile.index("\nup:\n") : makefile.index("\ndown:", makefile.index("\nup:\n"))]
    assert "physical-up" in up_block
    assert "physical-ready" not in up_block
    assert "tenant-create:" in makefile
    assert "tenant-sync:" in makefile
    assert "tenant-rotate:" in makefile
    assert "tenant-revoke:" in makefile
    assert "tenant-status:" in makefile
    assert "ready: physical-ready" in makefile


def test_local_secret_store_has_no_environment_fallback() -> None:
    source = _text("src/rag_ops_guard/local_credentials.py")

    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "RAG_OPS_API_KEY" not in source
    assert "secret-tool" in source
