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
UNREACHABLE_BUDGET_LINES = 2105  # U1a target; may only decrease in later units.
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


def test_core_does_not_import_langchain_or_langgraph() -> None:
    """SPEC-1a.4: framework dependencies belong to adapters, not the core."""
    files = _module_files()
    source = SRC / "agent/conversation.py"
    imports = _imports("rag_ops_guard.agent.conversation", source, files)
    leaked = sorted(item for item in imports if item.startswith(("langchain", "langgraph")))
    assert not leaked, f"framework imports leaked into core: {leaked}"


def test_unreachable_code_only_shrinks() -> None:
    """R-2/R-3: U1a must reduce the unreachable pipeline budget to <=2105 lines."""
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
    assert total <= UNREACHABLE_BUDGET_LINES, (
        f"unreachable pipeline code is {total} lines; budget is {UNREACHABLE_BUDGET_LINES}: "
        f"{sorted(dead)}"
    )
