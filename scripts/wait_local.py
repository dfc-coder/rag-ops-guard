from __future__ import annotations

import os
import subprocess
import time

import httpx

from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter


def env_port(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


FLOCI_HOST_PORT = env_port("FLOCI_HOST_PORT", 4566)
LLAMA_GEN_HOST_PORT = env_port("LLAMA_GEN_HOST_PORT", 8080)
LLAMA_EMBED_HOST_PORT = env_port("LLAMA_EMBED_HOST_PORT", 8081)
LLAMA_RERANK_HOST_PORT = env_port("LLAMA_RERANK_HOST_PORT", 8082)

FLOCI_CONTAINER = os.environ.get("FLOCI_CONTAINER_NAME", "rag-ops-floci")
LLAMA_GEN_CONTAINER = os.environ.get("LLAMA_GEN_CONTAINER_NAME", "rag-ops-llama-gen")
LLAMA_EMBED_CONTAINER = os.environ.get("LLAMA_EMBED_CONTAINER_NAME", "rag-ops-llama-embed")
LLAMA_RERANK_CONTAINER = os.environ.get("LLAMA_RERANK_CONTAINER_NAME", "rag-ops-llama-rerank")

SERVICES = {
    "floci": (f"http://127.0.0.1:{FLOCI_HOST_PORT}/", FLOCI_CONTAINER),
    "llama-gen": (f"http://127.0.0.1:{LLAMA_GEN_HOST_PORT}/health", LLAMA_GEN_CONTAINER),
    "llama-embed": (
        f"http://127.0.0.1:{LLAMA_EMBED_HOST_PORT}/health",
        LLAMA_EMBED_CONTAINER,
    ),
    "llama-rerank": (
        f"http://127.0.0.1:{LLAMA_RERANK_HOST_PORT}/health",
        LLAMA_RERANK_CONTAINER,
    ),
}


def container_logs(container: str) -> str:
    result = subprocess.run(
        ["podman", "logs", "--tail", "40", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout + result.stderr).strip()


def container_running(container: str) -> bool:
    result = subprocess.run(
        ["podman", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def wait_for(name: str, url: str, container: str, timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not started"
    while time.monotonic() < deadline:
        if not container_running(container):
            logs = container_logs(container)
            raise SystemExit(f"{name} container stopped during startup:\n{logs}")
        try:
            response = httpx.get(url, timeout=3.0)
            if response.status_code < 500:
                print(f"{name}: ready ({response.status_code})")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2)
    logs = container_logs(container)
    raise SystemExit(f"{name} did not become ready: {last_error}\n{logs}")


def verify_reranker() -> None:
    base_url = os.environ.get(
        "RERANKER_BASE_URL",
        f"http://127.0.0.1:{LLAMA_RERANK_HOST_PORT}",
    )
    adapter = LlamaCppRerankerAdapter(
        base_url,
        "qwen3-reranker-0.6b",
        timeout_seconds=30.0,
    )
    try:
        grades = adapter.grade(
            "How many retries does Calypso allow?",
            [
                "Calypso retry policy: automated processing stops after three retries.",
                "Kafka consumer groups may retry message delivery.",
            ],
        )
        if len(grades) != 2 or not grades[0].relevant or grades[1].relevant:
            raise ValueError(f"unexpected relevance grades: {grades}")
    except (httpx.HTTPError, ValueError) as exc:
        logs = container_logs(LLAMA_RERANK_CONTAINER)
        raise SystemExit(f"llama-rerank capability probe failed: {exc}\n{logs}") from exc
    print("llama-rerank: functional yes/no grading")


def main() -> None:
    for name, (url, container) in SERVICES.items():
        wait_for(name, url, container)
    verify_reranker()


if __name__ == "__main__":
    main()
