from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from rag_ops_guard.config import Settings

JUDGE_SYSTEM_PROMPT = (
    "Judge only the provided question, response, reference, and contexts. "
    "Do not use external knowledge."
)
DEFAULT_DATASET_PATHS = (
    Path("evaluation/datasets/golden-v1.json"),
    Path("evaluation/datasets/golden-mixed-v1.json"),
)


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


def _normalize_dataset_paths(dataset_paths: Path | tuple[Path, ...]) -> tuple[Path, ...]:
    return (dataset_paths,) if isinstance(dataset_paths, Path) else dataset_paths


def evaluation_dataset_sha256(
    dataset_paths: Path | tuple[Path, ...] = DEFAULT_DATASET_PATHS,
) -> str:
    paths = _normalize_dataset_paths(dataset_paths)
    if not paths:
        raise SystemExit("judge identity requires at least one evaluation dataset")
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"evaluation dataset not found: {path}")
        encoded_name = path.as_posix().encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def judge_identity_from_settings(
    settings: Settings,
    dataset_paths: Path | tuple[Path, ...] = DEFAULT_DATASET_PATHS,
) -> JudgeIdentity:
    return JudgeIdentity(
        provider=settings.ragas_judge_provider,
        model=settings.resolved_ragas_judge_model,
        prompt_sha256=_sha256_bytes(JUDGE_SYSTEM_PROMPT.encode("utf-8")),
        dataset_sha256=evaluation_dataset_sha256(dataset_paths),
    )


def judge_connection_from_settings(
    settings: Settings,
    dataset_paths: Path | tuple[Path, ...] = DEFAULT_DATASET_PATHS,
) -> JudgeConnection:
    identity = judge_identity_from_settings(settings, dataset_paths)
    api_key = (
        settings.ragas_judge_api_key.get_secret_value()
        if settings.ragas_judge_api_key is not None
        else "local"
    )
    return JudgeConnection(
        identity=identity,
        base_url=settings.resolved_ragas_judge_base_url.rstrip("/"),
        api_key=api_key,
    )
