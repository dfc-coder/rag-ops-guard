from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from rag_ops_guard.agent.semantic_router import (
    SemanticConversationContext,
    SemanticTurnResolver,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/evals/semantic_turn_cases.jsonl"


def _context(kind: str) -> SemanticConversationContext:
    if kind == "grounded":
        return SemanticConversationContext(
            has_grounded_topic=True,
            grounded_topic="payments: Calypso retry policy",
            last_grounded_query="Cuantos reintentos permite Calypso?",
            has_active_evidence=True,
            last_user_message="Cuantos reintentos permite Calypso?",
            last_assistant_message="Calypso permite tres reintentos automaticos.",
            system_filter=None,
            environment_filter="production",
            api_version_filter=None,
        )
    return SemanticConversationContext(
        has_grounded_topic=False,
        grounded_topic=None,
        last_grounded_query=None,
        has_active_evidence=False,
        last_user_message=None,
        last_assistant_message=None,
        system_filter=None,
        environment_filter=None,
        api_version_filter=None,
    )


def _accepted(expected: Any, actual: str) -> bool:
    if isinstance(expected, str):
        return actual == expected
    if isinstance(expected, list):
        return actual in {str(item) for item in expected}
    return False


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    resolver = SemanticTurnResolver.from_settings()
    failures: list[str] = []
    latencies_ms: list[float] = []

    for raw_line in CASES.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        case = json.loads(raw_line)
        started = perf_counter()
        decision = resolver.resolve(case["message"], _context(case["context"]))
        elapsed_ms = (perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)
        print(
            f"{case['name']}: {elapsed_ms:.0f}ms grounding={decision.requires_grounding} "
            f"relation={decision.relation_to_context.value} "
            f"operation={decision.operation.value} query={decision.standalone_query!r}"
        )

        if decision.requires_grounding is not case["requires_grounding"]:
            failures.append(f"{case['name']}: requires_grounding")
        if not _accepted(case["relation_to_context"], decision.relation_to_context.value):
            failures.append(f"{case['name']}: relation_to_context")
        if decision.operation.value != case["operation"]:
            failures.append(f"{case['name']}: operation")
        if decision.requires_grounding and not (decision.standalone_query or "").strip():
            failures.append(f"{case['name']}: standalone_query")

    average = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    print(
        "semantic-router latency: "
        f"avg={average:.0f}ms p50={_percentile(latencies_ms, 0.50):.0f}ms "
        f"p95={_percentile(latencies_ms, 0.95):.0f}ms max={max(latencies_ms, default=0):.0f}ms"
    )

    if failures:
        raise SystemExit("SEMANTIC ROUTER SMOKE FAILED: " + ", ".join(failures))

    print("SEMANTIC ROUTER V3 READY: structured semantic routing evaluation passed")


if __name__ == "__main__":
    main()
