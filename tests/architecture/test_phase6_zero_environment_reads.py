from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/rag_ops_guard"


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
        and node.attr == "environ"
    )


def _environment_reads() -> list[str]:
    violations: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "getenv"
                ):
                    violations.append(path.relative_to(SRC).as_posix())
                if node.func.attr == "get" and _is_os_environ(node.func.value):
                    violations.append(path.relative_to(SRC).as_posix())
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and _is_os_environ(node.value)
            ):
                violations.append(path.relative_to(SRC).as_posix())
    return sorted(set(violations))


def test_application_has_zero_process_environment_reads() -> None:
    assert _environment_reads() == []


def test_application_does_not_depend_on_dotenv_runtime_contract() -> None:
    for path in SRC.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "load_dotenv" not in source
        assert "dotenv_values" not in source
        assert "env_file" not in source
