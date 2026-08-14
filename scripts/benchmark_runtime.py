from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx


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


def run_one(url: str, question: str) -> tuple[float, dict[str, Any]]:
    started = time.perf_counter()
    response = httpx.post(
        f"{url}/v1/query",
        json={"question": question, "context": {}},
        timeout=180,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return elapsed_ms, dict(response.json())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--question", default="¿Qué puedes contarme de Calypso?")
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be >= 1")

    url = api_url()
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda _: run_one(url, args.question),
                range(args.requests),
            )
        )
    wall_seconds = time.perf_counter() - wall_started

    latencies = [elapsed for elapsed, _ in results]
    statuses = Counter(str(payload.get("status")) for _, payload in results)
    generation = [
        float(payload.get("timings_ms", {}).get("generation", 0.0)) for _, payload in results
    ]

    print(f"requests={args.requests} concurrency={args.concurrency}")
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


if __name__ == "__main__":
    main()
