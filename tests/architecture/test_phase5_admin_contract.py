from __future__ import annotations

import ast
from pathlib import Path

from rag_ops_guard.configstore.registry import registry_entries

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/rag_ops_guard"

# Phase 5 is allowed to preserve the Phase-4 bootstrap debt, but it may not add to it.
PHASE4_ENV_READ_FILES = {
    "configstore/runtime.py",
    "configstore/tenant_runtime.py",
    "handlers/ingest.py",
    "handlers/query.py",
}
PHASE4_LAMBDA_ENV_KEYS = {
    "S3_DOCUMENT_BUCKET",
    "S3_VECTOR_BUCKET",
    "S3_VECTOR_INDEX",
    "VECTOR_DIMENSION",
    "CONFIG_SOURCE",
    "CONFIG_TABLE",
    "TENANT_TABLE",
    "CONFIG_HASH",
    "CONFIG_HEAD_TTL_SECONDS",
    "CONFIG_MAX_STALE_SECONDS",
    "APP_ENV",
    "AWS_ENDPOINT_URL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSION",
    "RERANKER_BASE_URL",
    "RERANKER_MODEL",
}


def _env_read_files() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    if node.func.attr in {"getenv"}:
                        found.add(path.relative_to(SRC).as_posix())
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
                if (
                    isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "os"
                    and node.value.attr == "environ"
                ):
                    found.add(path.relative_to(SRC).as_posix())
    return found


def test_phase5_adds_no_new_application_environment_reads() -> None:
    assert _env_read_files() <= PHASE4_ENV_READ_FILES


def test_phase5_freezes_lambda_application_environment_until_phase6() -> None:
    stack = (ROOT / "infra/cdk/lib/rag-ops-guard-stack.ts").read_text(encoding="utf-8")
    start = stack.index("const commonEnvironment = {")
    end = stack.index("};", start)
    block = stack[start:end]
    present = {
        key
        for key in PHASE4_LAMBDA_ENV_KEYS
        if f"{key}:" in block
    }
    assert present == PHASE4_LAMBDA_ENV_KEYS
    uppercase_keys = {
        line.strip().split(":", 1)[0]
        for line in block.splitlines()[1:]
        if ":" in line and line.strip() and line.strip()[0].isupper()
    }
    assert uppercase_keys == PHASE4_LAMBDA_ENV_KEYS


def test_phase5_admin_domain_does_not_depend_on_concrete_phase3_store() -> None:
    source = (SRC / "configstore/admin.py").read_text(encoding="utf-8")
    assert "DynamoDbSecretStore" not in source
    assert "EnvelopeSecretService" not in source
    assert "secret_store" not in source
    assert "secret_service" not in source
    assert "class SecretBackend(Protocol)" in source


def test_secret_fields_remain_excluded_from_ordinary_config_publication() -> None:
    entries = registry_entries()
    secret_names = {name for name, entry in entries.items() if entry.sensitivity == "secret"}
    assert secret_names
    publisher = (ROOT / "scripts/config_publish.py").read_text(encoding="utf-8")
    assert "entry.sensitivity == \"secret\"" in publisher


def test_phase6_spec_exists_but_no_phase6_runtime_is_started() -> None:
    assert (ROOT / "docs/specs/phase6-environmentless-control-plane.md").is_file()
    assert not (SRC / "control_plane").exists()
