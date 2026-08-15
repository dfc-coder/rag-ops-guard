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
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)) or not node.elts:
        return False
    return all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in node.elts
    )


def _regex_compile_calls(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr == "compile":
            owner = function.value
            if isinstance(owner, ast.Name) and owner.id == "re":
                lines.append(node.lineno)
    return lines


def _routing_modules() -> list[Path]:
    guard = _load_guard()
    modules = guard["routing_modules"]
    assert isinstance(modules, list)
    assert all(isinstance(module, str) for module in modules)
    return [ROOT / str(module) for module in modules]


def test_production_routing_has_zero_dialogue_catalogs() -> None:
    guard = _load_guard()
    assert guard["max_legacy_string_literals"] == 0

    offenders: list[str] = []
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, value in _top_level_assignments(tree).items():
            if _direct_string_collection(value):
                offenders.append(f"{path.relative_to(ROOT)}:{name}")

    assert not offenders, (
        "Dialogue/entity string catalogs are forbidden in production routing code. "
        f"Move language examples to eval data instead: {offenders}"
    )


def test_production_routing_has_zero_regex_dialogue_router() -> None:
    guard = _load_guard()
    assert guard["max_legacy_regex_patterns"] == 0

    offenders: list[str] = []
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for line in _regex_compile_calls(tree):
            offenders.append(f"{path.relative_to(ROOT)}:{line}")

    assert not offenders, (
        "Regex routing is forbidden in production semantic routing modules. "
        f"Use the structured semantic resolver instead: {offenders}"
    )


def test_guard_declares_semantic_only_zero_heuristic_state() -> None:
    guard = _load_guard()
    assert guard["policy"] == "semantic-routing-only"
    assert guard["target_legacy_string_literals"] == 0
    assert guard["target_legacy_regex_patterns"] == 0
    assert guard["max_legacy_string_literals"] == 0
    assert guard["max_legacy_regex_patterns"] == 0
