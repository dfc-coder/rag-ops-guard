from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

from rag_ops_guard.agent.semantic_gate import (
    GroundingAction,
    SemanticGateContext,
    SemanticGroundingGate,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/evals/semantic_gate_cases.jsonl"


def _context(kind: str) -> SemanticGateContext:
    grounded = kind == "grounded"
    return SemanticGateContext(
        has_grounded_context=grounded,
        has_active_evidence=grounded,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _accepted(expected: str, actual: GroundingAction) -> bool:
    if expected == GroundingAction.RETRIEVE.value:
        return actual in {GroundingAction.RETRIEVE, GroundingAction.UNCERTAIN}
    return actual.value == expected


def main() -> None:
    gate = SemanticGroundingGate.from_settings()
    failures: list[str] = []
    latencies_ms: list[float] = []
    uncertain = 0

    for raw_line in CASES.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        case = json.loads(raw_line)
        started = perf_counter()
        decision = gate.decide(case["message"], _context(case["context"]))
        elapsed_ms = (perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)
        uncertain += int(decision.action == GroundingAction.UNCERTAIN)
        print(
            f"{case['name']}: {elapsed_ms:.0f}ms action={decision.action.value} "
            f"score={decision.score:.4f} margin={decision.margin:.4f} "
            f"scores={decision.scores}"
        )
        if not _accepted(case["expected"], decision.action):
            failures.append(
                f"{case['name']}: expected={case['expected']} actual={decision.action.value}"
            )

    average = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50 = _percentile(latencies_ms, 0.50)
    p95 = _percentile(latencies_ms, 0.95)
    maximum = max(latencies_ms, default=0.0)
    print(
        "cross-encoder-gate latency: "
        f"avg={average:.0f}ms p50={p50:.0f}ms p95={p95:.0f}ms "
        f"max={maximum:.0f}ms uncertain={uncertain}"
    )

    p95_limit = float(os.getenv("SEMANTIC_GATE_P95_MAX_MS", "1500"))
    if p95 > p95_limit:
        failures.append(f"p95 latency {p95:.0f}ms exceeds {p95_limit:.0f}ms")

    if failures:
        raise SystemExit("CROSS-ENCODER GATE FAILED: " + ", ".join(failures))

    print("CROSS-ENCODER GATE V5 READY: learned policy scoring passed")


if __name__ == "__main__":
    main()
