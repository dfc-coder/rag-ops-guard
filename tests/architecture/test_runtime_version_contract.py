from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_python_runtime_has_one_project_source_of_truth() -> None:
    version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    package_script = (ROOT / "scripts/package_lambda.sh").read_text(encoding="utf-8")
    cdk_bin = (ROOT / "infra/cdk/bin/rag-ops-guard.ts").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert version == "3.13"
    assert pyproject["project"]["requires-python"] == ">=3.13,<3.14"
    assert pyproject["tool"]["ruff"]["target-version"] == "py313"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.13"

    assert "PYTHON_VERSION := $(strip $(shell cat .python-version))" in makefile
    assert "UV_PYTHON ?= $(PYTHON_VERSION)" in makefile
    assert "LAMBDA_PYTHON_VERSION ?= $(PYTHON_VERSION)" in makefile
    assert "--python $(PYTHON_VERSION)" in makefile

    assert 'PYTHON_VERSION_FILE="$ROOT/.python-version"' in package_script
    assert 'PROJECT_PYTHON_VERSION="$(tr -d' in package_script
    assert "3.12" not in package_script

    assert ".python-version" in cdk_bin
    assert "pythonVersion" in cdk_bin
    assert "python-version: '3.12'" not in ci
    assert "python-version: '3.13'" in ci
