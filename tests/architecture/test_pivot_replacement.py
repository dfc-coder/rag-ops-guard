from __future__ import annotations

import ast
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/rag_ops_guard"
CANONICAL_ROOTS = {
    "rag_ops_guard.app",
    "rag_ops_guard.agent.conversation",
    "rag_ops_guard.handlers.query",
}
LAMBDA_ENTRYPOINTS = {
    "rag_ops_guard.handlers.ingest",
    "rag_ops_guard.handlers.health",
}
PIPELINE_OWNERS = {"app", "agent.conversation"}
UNREACHABLE_BUDGET_LINES = 0
PIPELINE_DIRS = ("agent", "graph", "retrieval", "ports")
PIPELINE_PREFIXES = tuple(f"rag_ops_guard.{package}" for package in PIPELINE_DIRS)


def _module_for(path: Path) -> str:
    relative = path.relative_to(ROOT / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for package in PIPELINE_DIRS:
        base = SRC / package
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            files[_module_for(path)] = path
    for path in (SRC / "handlers").glob("*.py"):
        files[_module_for(path)] = path
    files["rag_ops_guard.app"] = SRC / "app.py"
    return files


def _resolve_from(module: str, node: ast.ImportFrom, files: dict[str, Path]) -> str | None:
    if node.level == 0:
        return node.module
    current = module.split(".")
    path = files.get(module)
    if path is not None and path.name != "__init__.py":
        current = current[:-1]
    if node.level > len(current):
        return None
    prefix = current[: len(current) - node.level + 1]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _imports(module: str, path: Path, files: dict[str, Path]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from(module, node, files)
            if resolved:
                result.add(resolved)
    return result


def _reachable() -> set[str]:
    files = _module_files()
    reachable: set[str] = set()
    queue = deque(CANONICAL_ROOTS | LAMBDA_ENTRYPOINTS)
    while queue:
        module = queue.popleft()
        if module in reachable:
            continue
        reachable.add(module)
        path = files.get(module)
        if path is None:
            continue
        for imported in _imports(module, path, files):
            queue.extend(
                candidate
                for candidate in files
                if candidate == imported or candidate.startswith(imported + ".")
            )
    return reachable


def test_final_pipeline_owners_are_only_app_and_conversation_agent() -> None:
    """SPEC-6 / R-3 + SPEC-P4-MULTITENANCY 4.2."""
    assert PIPELINE_OWNERS == {"app", "agent.conversation"}
    app_tree = ast.parse((SRC / "app.py").read_text(encoding="utf-8"))
    conversation = (SRC / "agent/conversation.py").read_text(encoding="utf-8")
    functions = {
        node.name: node for node in app_tree.body if isinstance(node, ast.FunctionDef)
    }
    canonical = functions["conversation_agent"]
    assert canonical.args.args[0].arg == "context"
    assert canonical.args.defaults
    assert isinstance(canonical.args.defaults[0], ast.Constant)
    assert canonical.args.defaults[0].value is None
    assert "class ConversationAgent" in conversation


def test_graph_package_is_physically_removed() -> None:
    """SPEC-6 / R-1"""
    assert not (SRC / "graph").exists()


def test_core_does_not_import_langchain_or_langgraph() -> None:
    """SPEC-1a.4"""
    files = _module_files()
    source = SRC / "agent/conversation.py"
    imports = _imports("rag_ops_guard.agent.conversation", source, files)
    leaked = sorted(item for item in imports if item.startswith(("langchain", "langgraph")))
    assert not leaked, f"framework imports leaked into core: {leaked}"


def test_unreachable_pipeline_code_is_zero() -> None:
    """SPEC-6 / R-2 / R-3"""
    files = _module_files()
    reachable = _reachable()
    dead = {
        module: path
        for module, path in files.items()
        if module.startswith(PIPELINE_PREFIXES)
        and module not in reachable
        and module not in LAMBDA_ENTRYPOINTS
    }
    total = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in dead.values())
    assert total == UNREACHABLE_BUDGET_LINES, (
        f"unreachable pipeline code is {total} lines; expected zero: {sorted(dead)}"
    )


def test_citation_validator_is_on_the_canonical_path() -> None:
    """SPEC-1.2"""
    reachable = _reachable()
    assert "rag_ops_guard.retrieval.citations" in reachable
