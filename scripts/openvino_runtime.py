from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

OVMS_IMAGE = os.environ.get("OVMS_IMAGE", "docker.io/openvino/model_server:latest-gpu")
OVMS_CONTAINER_NAME = os.environ.get("OVMS_CONTAINER_NAME", "rag-ops-ovms-rag")
OVMS_HOST_PORT = int(os.environ.get("OVMS_HOST_PORT", "8083"))
EMBEDDING_MODEL = os.environ.get(
    "OVMS_EMBEDDING_MODEL",
    "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
)
RERANKER_MODEL = os.environ.get(
    "OVMS_RERANKER_MODEL",
    "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov",
)


def model_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return Path(
        os.environ.get("OVMS_MODEL_DIR", cache_home / "rag-ops-guard" / "openvino-models")
    ).expanduser().resolve()


def render_group_id() -> str:
    render_nodes = sorted(Path("/dev/dri").glob("render*"))
    if not render_nodes:
        raise SystemExit("No /dev/dri/render* device found; Intel GPU is unavailable.")
    return str(render_nodes[0].stat().st_gid)


def podman(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["podman", *args],
        check=check,
        text=True,
        capture_output=not check,
    )


def stop() -> None:
    podman("rm", "-f", OVMS_CONTAINER_NAME, check=False)


def wait_ready(timeout_seconds: int = 180) -> None:
    base = f"http://127.0.0.1:{OVMS_HOST_PORT}/v3"
    deadline = time.monotonic() + timeout_seconds
    last_error = "not started"
    while time.monotonic() < deadline:
        try:
            embed = httpx.post(
                f"{base}/embeddings",
                json={"model": EMBEDDING_MODEL, "input": "hardware acceleration probe"},
                timeout=20.0,
            )
            rerank = httpx.post(
                f"{base}/rerank",
                json={
                    "model": RERANKER_MODEL,
                    "query": "Calypso retries",
                    "documents": [
                        "Calypso supports three automated retries.",
                        "The cafeteria opens at nine.",
                    ],
                    "top_n": 2,
                },
                timeout=20.0,
            )
            if embed.is_success and rerank.is_success:
                print(f"OpenVINO RAG backend: ready on {base}")
                return
            last_error = (
                f"embeddings={embed.status_code} {embed.text[:160]!r}; "
                f"rerank={rerank.status_code} {rerank.text[:160]!r}"
            )
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2)

    logs = podman("logs", "--tail", "80", OVMS_CONTAINER_NAME, check=False)
    raise SystemExit(
        f"OpenVINO backend did not become ready: {last_error}\n"
        f"{logs.stdout}{logs.stderr}"
    )


def start() -> None:
    directory = model_dir()
    config = directory / "config.json"
    if not config.exists():
        raise SystemExit("OpenVINO config.json is missing. Run `make openvino-models` first.")

    stop()
    command = [
        "run",
        "-d",
        "--name",
        OVMS_CONTAINER_NAME,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--device",
        "/dev/dri",
        "--group-add",
        render_group_id(),
        "-p",
        f"127.0.0.1:{OVMS_HOST_PORT}:8000",
        "-v",
        f"{directory}:/models:ro,Z",
        OVMS_IMAGE,
        "--rest_port",
        "8000",
        "--config_path",
        "/models/config.json",
    ]
    subprocess.run(["podman", *command], check=True)
    wait_ready()


def status() -> None:
    result = podman("inspect", "-f", "{{.State.Running}}", OVMS_CONTAINER_NAME, check=False)
    running = result.returncode == 0 and result.stdout.strip() == "true"
    print(f"{OVMS_CONTAINER_NAME}: {'running' if running else 'stopped'}")
    if running:
        print(f"http://127.0.0.1:{OVMS_HOST_PORT}/v3")


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "up":
        start()
    elif action == "down":
        stop()
    elif action == "status":
        status()
    elif action == "wait":
        wait_ready()
    else:
        raise SystemExit("usage: openvino_runtime.py [up|down|status|wait]")


if __name__ == "__main__":
    main()
