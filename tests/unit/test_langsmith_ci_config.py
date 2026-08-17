from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module() -> ModuleType:
    path = ROOT / "scripts/local/configure_langsmith_ci.py"
    spec = importlib.util.spec_from_file_location("rag_ops_langsmith_ci", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_langsmith_env_parser_reads_only_supported_keys(tmp_path: Path) -> None:
    module = _load_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LANGSMITH_API_KEY='key-1'\n"
        "LANGSMITH_PROJECT=project-1\n"
        "AWS_SECRET_ACCESS_KEY=must-not-load\n",
        encoding="utf-8",
    )

    assert module._parse_env_file(env_file) == {
        "LANGSMITH_API_KEY": "key-1",
        "LANGSMITH_PROJECT": "project-1",
    }


def test_langsmith_ci_writes_masked_required_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    source = tmp_path / "langsmith.env"
    source.write_text("LANGSMITH_API_KEY=local-key\n", encoding="utf-8")
    target = tmp_path / "github-env"
    monkeypatch.setenv("RAG_OPS_LANGSMITH_ENV_FILE", str(source))
    for key in module.LANGSMITH_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("REPO_LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("REPO_LANGSMITH_WORKSPACE_ID", raising=False)

    values = module.resolve_langsmith_environment()
    module.write_github_environment(target, values)

    written = target.read_text(encoding="utf-8")
    assert "LANGSMITH_TRACING=true" in written
    assert "LANGSMITH_PROJECT=rag-ops-guard-physical-golden" in written
    assert "LANGSMITH_API_KEY=local-key" in written
    assert "::add-mask::local-key" in capsys.readouterr().out


def test_langsmith_ci_fails_without_any_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_candidate_files", lambda: [])
    for key in module.LANGSMITH_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("REPO_LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("REPO_LANGSMITH_WORKSPACE_ID", raising=False)

    with pytest.raises(SystemExit, match="LANGSMITH_API_KEY"):
        module.write_github_environment(tmp_path / "github-env", module.resolve_langsmith_environment())
