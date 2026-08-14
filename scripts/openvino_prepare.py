from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

OVMS_IMAGE = os.environ.get("OVMS_IMAGE", "docker.io/openvino/model_server:latest-gpu")
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
        raise SystemExit(
            "OpenVINO GPU backend requires /dev/dri/render*. "
            "The Intel GPU is not exposed to this Linux session."
        )
    return str(render_nodes[0].stat().st_gid)


def run_ovms(args: list[str], *, gpu: bool) -> None:
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        "podman",
        "run",
        "--rm",
        # Rootless Podman normally maps the host user to container root. keep-id makes
        # the bind-mounted cache owned by the same numeric UID inside the container,
        # so OVMS can create /models/OpenVINO without chowning host files.
        "--userns=keep-id",
        "-v",
        f"{directory}:/models:rw,Z",
    ]
    if gpu:
        command.extend(["--device", "/dev/dri"])
    command.extend([OVMS_IMAGE, *args])
    subprocess.run(command, check=True)


def configured_models() -> set[str]:
    config_path = model_dir() / "config.json"
    if not config_path.exists():
        return set()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    configured: set[str] = set()
    for item in payload.get("model_config_list", []):
        config = item.get("config", {}) if isinstance(item, dict) else {}
        name = config.get("name")
        if isinstance(name, str):
            configured.add(name)
    for item in payload.get("mediapipe_config_list", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str):
            configured.add(name)
    return configured


def pull_model(model: str, task: str, *, pooling: str | None = None) -> None:
    args = [
        "--pull",
        "--source_model",
        model,
        "--model_repository_path",
        "/models",
        "--task",
        task,
        "--target_device",
        "GPU",
    ]
    if pooling:
        args.extend(["--pooling", pooling])
    run_ovms(args, gpu=True)


def add_to_config(model: str) -> None:
    run_ovms(
        [
            "--add_to_config",
            "--config_path",
            "/models/config.json",
            "--model_name",
            model,
            "--model_path",
            model,
        ],
        gpu=False,
    )


def ensure_model(model: str, task: str, *, pooling: str | None = None) -> None:
    if model in configured_models():
        print(f"OpenVINO model configured: {model}")
        return
    print(f"Preparing OpenVINO model on Intel GPU: {model}")
    pull_model(model, task, pooling=pooling)
    add_to_config(model)


def main() -> None:
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    print(f"OpenVINO model repository: {directory}")
    print(f"OpenVINO Model Server image: {OVMS_IMAGE}")
    print(f"render group: {render_group_id()}")

    ensure_model(EMBEDDING_MODEL, "embeddings", pooling="LAST")
    ensure_model(RERANKER_MODEL, "rerank")
    print("OpenVINO RAG models: ready")


if __name__ == "__main__":
    main()
