from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from rag_ops_guard.app import query_workflow
from rag_ops_guard.domain.models import QueryRequest


def api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run_one_direct(question: str) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    response = query_workflow().invoke(QueryRequest(question=question))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, dict(response.model_dump(mode="json"))


def run_one_api(url: str, question: str) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    response = httpx.post(
        f"{url}/v1/query",
        json={"question": question, "context": {}},
        timeout=180,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.is_error:
        raise RuntimeError(
            f"API benchmark failed with HTTP {response.status_code}: {response.text}"
        )
    return elapsed_ms, dict(response.json())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--transport", choices=["direct", "api"], default="direct")
    parser.add_argument("--question", default="¿Qué puedes contarme de Calypso?")
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be >= 1")

    url = api_url() if args.transport == "api" else ""
    wall_started = time.perf_counter()

    def run(_: int) -> tuple[float, dict[str, Any]]:
        started = time.perf_counter()
        try:
            if args.transport == "api":
                return run_one_api(url, args.question)
            return run_one_direct(args.question)
        except Exception as exc:  # benchmark must report failures, not stop the run
            elapsed_ms = (time.perf_counter() - started) * 1000
            return elapsed_ms, {
                "status": "error",
                "error": type(exc).__name__,
            }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, range(args.requests)))

    wall_seconds = time.perf_counter() - wall_started
    latencies = [elapsed for elapsed, _ in results]
    statuses = Counter(str(payload.get("status")) for _, payload in results)
    errors = Counter(
        str(payload.get("error"))
        for _, payload in results
        if payload.get("status") == "error"
    )
    generation = [
        float(payload.get("timings_ms", {}).get("generation", 0.0)) for _, payload in results
    ]

    print(f"transport={args.transport} requests={args.requests} concurrency={args.concurrency}")
    print(f"wall={wall_seconds:.2f}s throughput={args.requests / wall_seconds:.2f} req/s")
    print(
        "latency_ms "
        f"p50={percentile(latencies, 0.50):.0f} "
        f"p95={percentile(latencies, 0.95):.0f} "
        f"max={max(latencies):.0f}"
    )
    if any(generation):
        print(
            "generation_ms "
            f"p50={percentile(generation, 0.50):.0f} "
            f"p95={percentile(generation, 0.95):.0f}"
        )
    print("statuses=" + ", ".join(f"{key}:{value}" for key, value in sorted(statuses.items())))
    if errors:
        print("errors=" + ", ".join(f"{key}:{value}" for key, value in sorted(errors.items())))


if __name__ == "__main__":
    main()
