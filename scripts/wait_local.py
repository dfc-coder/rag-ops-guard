from __future__ import annotations

import subprocess
import time

import httpx

from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter

SERVICES = {
    "floci": ("http://127.0.0.1:4566/", "rag-ops-floci"),
    "llama-gen": ("http://127.0.0.1:8080/health", "rag-ops-llama-gen"),
    "llama-embed": ("http://127.0.0.1:8081/health", "rag-ops-llama-embed"),
    "llama-rerank": ("http://127.0.0.1:8082/health", "rag-ops-llama-rerank"),
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
    adapter = LlamaCppRerankerAdapter(
        "http://127.0.0.1:8082",
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
        logs = container_logs("rag-ops-llama-rerank")
        raise SystemExit(f"llama-rerank capability probe failed: {exc}\n{logs}") from exc
    print("llama-rerank: functional yes/no grading")


def main() -> None:
    for name, (url, container) in SERVICES.items():
        wait_for(name, url, container)
    verify_reranker()


if __name__ == "__main__":
    main()
