from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GROUNDING_MODULE = ROOT / "src/rag_ops_guard/agent/grounding.py"
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


def _direct_string_literals(node: ast.AST) -> int | None:
    if not isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return None
    count = 0
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            count += 1
        elif isinstance(element, ast.Starred):
            continue
        else:
            return None
    return count


def _regex_compile_count(node: ast.AST) -> int:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return 0
    count = 0
    for element in node.elts:
        if not isinstance(element, ast.Call):
            continue
        function = element.func
        if isinstance(function, ast.Attribute) and function.attr == "compile":
            count += 1
    return count


def test_dialogue_heuristic_budget_can_only_decrease() -> None:
    guard = _load_guard()
    source = GROUNDING_MODULE.read_text(encoding="utf-8")
    assignments = _top_level_assignments(ast.parse(source))

    legacy = guard["legacy_catalogs"]
    assert isinstance(legacy, dict)

    observed_total = 0
    for name, maximum in legacy.items():
        assert isinstance(name, str)
        assert isinstance(maximum, int)
        value = assignments.get(name)
        if value is None:
            continue
        observed = _direct_string_literals(value)
        assert observed is not None, f"legacy catalogue {name} changed shape; remove it instead"
        assert observed <= maximum, (
            f"{name} grew from its frozen legacy budget {maximum} to {observed}. "
            "Do not add dialogue/entity heuristics; use the semantic structured resolver."
        )
        observed_total += observed

    maximum_total = guard["max_legacy_string_literals"]
    assert isinstance(maximum_total, int)
    assert observed_total <= maximum_total


def test_no_new_top_level_dialogue_catalogs() -> None:
    guard = _load_guard()
    legacy = guard["legacy_catalogs"]
    assert isinstance(legacy, dict)
    allowed_names = set(legacy)

    source = GROUNDING_MODULE.read_text(encoding="utf-8")
    assignments = _top_level_assignments(ast.parse(source))
    observed_catalogs = {
        name
        for name, value in assignments.items()
        if _direct_string_literals(value) is not None
    }

    unexpected = observed_catalogs - allowed_names
    assert not unexpected, (
        "New top-level string catalog(s) detected in grounding policy: "
        f"{sorted(unexpected)}. Language understanding belongs in the semantic resolver, "
        "not in Python phrase/entity lists."
    )


def test_regex_dialogue_router_budget_can_only_decrease() -> None:
    guard = _load_guard()
    legacy_regex = guard["legacy_regex_catalogs"]
    assert isinstance(legacy_regex, dict)

    source = GROUNDING_MODULE.read_text(encoding="utf-8")
    assignments = _top_level_assignments(ast.parse(source))
    observed_total = 0

    for name, maximum in legacy_regex.items():
        assert isinstance(name, str)
        assert isinstance(maximum, int)
        value = assignments.get(name)
        if value is None:
            continue
        observed = _regex_compile_count(value)
        assert observed <= maximum, (
            f"{name} grew from its frozen regex budget {maximum} to {observed}. "
            "Regex-based dialogue/entity routing is forbidden; use semantic structured output."
        )
        observed_total += observed

    maximum_total = guard["max_legacy_regex_patterns"]
    assert isinstance(maximum_total, int)
    assert observed_total <= maximum_total


def test_guard_declares_zero_heuristic_end_state() -> None:
    guard = _load_guard()
    assert guard["policy"] == "semantic-routing-only"
    assert guard["target_legacy_string_literals"] == 0
    assert guard["target_legacy_regex_patterns"] == 0
