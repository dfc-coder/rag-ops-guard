from __future__ import annotations

import subprocess
import time

import httpx

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
    try:
        response = httpx.post(
            "http://127.0.0.1:8082/v1/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": "payment retry policy",
                "documents": [
                    "Payment Retry Policy: transient payment failures may be retried.",
                    "Weather forecast for tomorrow.",
                ],
                "top_n": 2,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 2:
            raise ValueError("reranker probe did not return two scored documents")
        indexes = {item.get("index") for item in results if isinstance(item, dict)}
        if indexes != {0, 1}:
            raise ValueError("reranker probe returned invalid document indexes")
    except (httpx.HTTPError, ValueError) as exc:
        logs = container_logs("rag-ops-llama-rerank")
        raise SystemExit(f"llama-rerank capability probe failed: {exc}\n{logs}") from exc
    print("llama-rerank: functional")


def main() -> None:
    for name, (url, container) in SERVICES.items():
        wait_for(name, url, container)
    verify_reranker()


if __name__ == "__main__":
    main()
