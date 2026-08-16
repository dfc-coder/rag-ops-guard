from __future__ import annotations

import ast
from json import loads
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD_FILE = ROOT / "architecture/grounding-policy-guard.json"


def _load_guard() -> dict[str, object]:
    return loads(GUARD_FILE.read_text(encoding="utf-8"))


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
    if isinstance(node, ast.Dict) and node.values:
        values = [value for value in node.values if value is not None]
        return bool(values) and all(
            isinstance(value, ast.Constant) and isinstance(value.value, str) for value in values
        )
    return False


def _direct_string_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


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


def _allowed_semantic_maps() -> set[str]:
    guard = _load_guard()
    values = guard["allowed_semantic_maps"]
    assert isinstance(values, list)
    assert all(isinstance(value, str) for value in values)
    return {str(value) for value in values}


def test_production_routing_has_zero_dialogue_catalogs_anywhere() -> None:
    guard = _load_guard()
    assert guard["max_legacy_string_literals"] == 0
    allowed_maps = _allowed_semantic_maps()

    offenders: list[str] = []
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        allowed_nodes = {id(value) for name, value in assignments.items() if name in allowed_maps}
        for node in ast.walk(tree):
            if _direct_string_collection(node) and id(node) not in allowed_nodes:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert not offenders, (
        "Dialogue/entity string catalogs or maps are forbidden in production routing code. "
        f"Move language examples to eval data instead: {offenders}"
    )


def test_allowed_semantic_map_has_one_abstract_hypothesis_per_action() -> None:
    allowed_maps = _allowed_semantic_maps()
    assert allowed_maps == {"POLICY_HYPOTHESES"}

    matches = 0
    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        value = assignments.get("POLICY_HYPOTHESES")
        if value is None:
            continue
        matches += 1
        assert isinstance(value, ast.Dict)
        assert len(value.values) == 3
        assert all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.values
        )

    assert matches == 1


def test_production_routing_has_no_hidden_phrase_constants() -> None:
    allowed = _allowed_string_constants()
    offenders: list[str] = []

    for path in _routing_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, value in _top_level_assignments(tree).items():
            if _direct_string_constant(value) and name not in allowed:
                offenders.append(f"{path.relative_to(ROOT)}:{name}")

    assert not offenders, (
        "Top-level routing phrase constants are forbidden. "
        f"Move task semantics into the approved semantic map: {offenders}"
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
        f"Use learned semantic routing instead: {offenders}"
    )


def test_guard_declares_zero_heuristic_learned_gate_state() -> None:
    guard = _load_guard()
    assert guard["policy"] == "learned-cross-encoder-gate"
    assert guard["target_legacy_string_literals"] == 0
    assert guard["target_legacy_regex_patterns"] == 0
    assert guard["max_legacy_string_literals"] == 0
    assert guard["max_legacy_regex_patterns"] == 0
