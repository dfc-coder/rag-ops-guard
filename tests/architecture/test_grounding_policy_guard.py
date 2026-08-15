from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUARD_FILE = ROOT / "architecture/grounding-policy-guard.json"


def _load_guard() -> dict[str, object]:
    return json.loads(GUARD_FILE.read_text(encoding="utf-8"))


def _top_level_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = value
    return assignments


def _direct_string_collection(node: ast.AST) -> bool:
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)) and node.elts:
        return all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in node.elts
        )
    if isinstance(node, ast.Dict) and node.keys:
        nodes = [item for item in (*node.keys, *node.values) if item is not None]
        return bool(nodes) and all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in nodes
        )
    return False


def _direct_string_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_collection_lines(tree: ast.AST) -> list[int]:
    return [node.lineno for node in ast.walk(tree) if _direct_string_collection(node)]


def _regex_usage_lines(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "re" for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "re":
            lines.append(node.lineno)
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                owner = function.value
                if isinstance(owner, ast.Name) and owner.id == "re":
                    lines.append(node.lineno)
    return sorted(set(lines))


def _routing_modules() -> list[Path]:
    guard = _load_guard()
    modules = guard["routing_modules"]
    assert isinstance(modules, list)
    assert all(isinstance(module, str) for module in modules)
    return [ROOT / str(module) for module in modules]


def _allowed_string_constants() -> set[str]:
    guard = _load_guard()
    values = guard["allowed_top_level_string_constants"]
    assert isinstance(values, list)
    assert all(isinstance(value, str) for value in values)
    return {str(value) for value in values}


def test_production_routing_has_zero_dialogue_catalogs_anywhere() -> None:
    guard = _load_guard()
    assert guard["max_legacy_string_literals"] == 0

    offenders: list[str] = []
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _string_collection_lines(tree):
            offenders.append(f"{path.relative_to(ROOT)}:{line}")

    assert not offenders, (
        "Dialogue/entity string catalogs or maps are forbidden anywhere in production routing "
        f"code. Move language examples to eval data instead: {offenders}"
    )


def test_production_routing_has_no_hidden_phrase_constants() -> None:
    allowed = _allowed_string_constants()
    offenders: list[str] = []

    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, value in _top_level_assignments(tree).items():
            if _direct_string_constant(value) and name not in allowed:
                offenders.append(f"{path.relative_to(ROOT)}:{name}")

    assert not offenders, (
        "Top-level routing phrase constants are forbidden. The semantic router prompt is the only "
        f"approved routing string constant: {offenders}"
    )


def test_production_routing_has_zero_regex_dependency() -> None:
    guard = _load_guard()
    assert guard["max_legacy_regex_patterns"] == 0

    offenders: list[str] = []
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _regex_usage_lines(tree):
            offenders.append(f"{path.relative_to(ROOT)}:{line}")

    assert not offenders, (
        "Regex is forbidden in production semantic routing modules. "
        f"Use structured semantic routing instead: {offenders}"
    )


def test_guard_declares_semantic_only_zero_heuristic_state() -> None:
    guard = _load_guard()
    assert guard["policy"] == "semantic-routing-only"
    assert guard["target_legacy_string_literals"] == 0
    assert guard["target_legacy_regex_patterns"] == 0
    assert guard["max_legacy_string_literals"] == 0
    assert guard["max_legacy_regex_patterns"] == 0
