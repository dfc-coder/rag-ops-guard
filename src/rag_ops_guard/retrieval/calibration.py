from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    min_positive: float
    max_negative: float
    separation: float


def derive_separating_threshold(
    positive_scores: list[float],
    negative_scores: list[float],
) -> CalibrationResult:
    """Derive a cutoff only when labeled positive/negative scores are separable."""
    if not positive_scores or not negative_scores:
        raise ValueError("calibration requires positive and negative scores")

    min_positive = min(positive_scores)
    max_negative = max(negative_scores)
    if max_negative >= min_positive:
        raise ValueError(
            "reranker calibration is not separable: "
            f"max_negative={max_negative:.6f} >= min_positive={min_positive:.6f}"
        )

    threshold = (max_negative + min_positive) / 2.0
    return CalibrationResult(
        threshold=threshold,
        min_positive=min_positive,
        max_negative=max_negative,
        separation=min_positive - max_negative,
    )


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_fingerprint(root: Path = Path("knowledge-base")) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.md") if path.is_file())
    if not files:
        raise ValueError(f"no markdown knowledge files found under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_calibrated_threshold(
    path: Path,
    *,
    reranker_model: str,
    corpus_root: Path = Path("knowledge-base"),
    dataset_path: Path = Path("evaluation/datasets/retrieval-calibration-v1.json"),
) -> float:
    if not path.exists():
        raise RuntimeError(
            f"missing reranker calibration artifact: {path}; run `make retrieval-calibrate`"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("reranker_model") != reranker_model:
        raise RuntimeError("reranker calibration model does not match runtime model")
    if payload.get("corpus_fingerprint") != corpus_fingerprint(corpus_root):
        raise RuntimeError("reranker calibration corpus fingerprint is stale")
    if payload.get("dataset_fingerprint") != file_fingerprint(dataset_path):
        raise RuntimeError("reranker calibration dataset fingerprint is stale")

    threshold = payload.get("threshold")
    if not isinstance(threshold, int | float):
        raise RuntimeError("reranker calibration artifact has no numeric threshold")
    return float(threshold)
