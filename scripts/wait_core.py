from __future__ import annotations

import os
import subprocess
import time

import httpx

FLOCI_HOST_PORT = int(os.environ.get("FLOCI_HOST_PORT", "4566"))
LLAMA_GEN_HOST_PORT = int(os.environ.get("LLAMA_GEN_HOST_PORT", "8080"))
FLOCI_CONTAINER = os.environ.get("FLOCI_CONTAINER_NAME", "rag-ops-floci")
LLAMA_GEN_CONTAINER = os.environ.get("LLAMA_GEN_CONTAINER_NAME", "rag-ops-llama-gen")

SERVICES = {
    "floci": (f"http://127.0.0.1:{FLOCI_HOST_PORT}/", FLOCI_CONTAINER),
    "llama-gen": (f"http://127.0.0.1:{LLAMA_GEN_HOST_PORT}/health", LLAMA_GEN_CONTAINER),
}


def container_logs(container: str) -> str:
    result = subprocess.run(
        ["podman", "logs", "--tail", "60", container],
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
            raise SystemExit(f"{name} stopped during startup:\n{container_logs(container)}")
        try:
            response = httpx.get(url, timeout=3.0)
            if response.status_code < 500:
                print(f"{name}: ready ({response.status_code})")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:160]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SystemExit(
        f"{name} did not become ready: {last_error}\n{container_logs(container)}"
    )


def main() -> None:
    for name, (url, container) in SERVICES.items():
        wait_for(name, url, container)


if __name__ == "__main__":
    main()
