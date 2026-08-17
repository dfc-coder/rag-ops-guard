from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

JUDGE_SYSTEM_PROMPT = (
    "Judge only the provided question, response, reference, and contexts. "
    "Do not use external knowledge."
)
DEFAULT_DATASET_PATH = Path("evaluation/datasets/golden-v1.json")
DEFAULT_RUNTIME_MODEL = "qwen3.5-0.8b-unsloth-ud-q4-k-xl"


@dataclass(frozen=True)
class JudgeIdentity:
    provider: str
    model: str
    prompt_sha256: str
    dataset_sha256: str


@dataclass(frozen=True)
class JudgeConnection:
    identity: JudgeIdentity
    base_url: str
    api_key: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def judge_identity_from_env(dataset_path: Path = DEFAULT_DATASET_PATH) -> JudgeIdentity:
    provider = os.environ.get("RAGAS_JUDGE_PROVIDER", "local").strip() or "local"
    runtime_model = os.environ.get("LLM_MODEL", DEFAULT_RUNTIME_MODEL).strip()
    model = os.environ.get("RAGAS_JUDGE_MODEL", runtime_model).strip()
    if not model:
        raise SystemExit("RAGAS_JUDGE_MODEL must not be empty")
    if not dataset_path.is_file():
        raise SystemExit(f"evaluation dataset not found: {dataset_path}")
    return JudgeIdentity(
        provider=provider,
        model=model,
        prompt_sha256=_sha256_bytes(JUDGE_SYSTEM_PROMPT.encode("utf-8")),
        dataset_sha256=_sha256_bytes(dataset_path.read_bytes()),
    )


def judge_connection_from_env(dataset_path: Path = DEFAULT_DATASET_PATH) -> JudgeConnection:
    identity = judge_identity_from_env(dataset_path)
    if identity.provider == "local":
        base_url = os.environ.get(
            "RAGAS_JUDGE_BASE_URL",
            os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        ).strip()
        api_key = os.environ.get("RAGAS_JUDGE_API_KEY", "local").strip() or "local"
    else:
        base_url = os.environ.get("RAGAS_JUDGE_BASE_URL", "").strip()
        api_key = os.environ.get("RAGAS_JUDGE_API_KEY", "").strip()
        if not base_url:
            raise SystemExit(
                "RAGAS_JUDGE_BASE_URL is required when RAGAS_JUDGE_PROVIDER is not 'local'"
            )
        if not api_key:
            raise SystemExit(
                "RAGAS_JUDGE_API_KEY is required when RAGAS_JUDGE_PROVIDER is not 'local'"
            )
    return JudgeConnection(identity=identity, base_url=base_url.rstrip("/"), api_key=api_key)
