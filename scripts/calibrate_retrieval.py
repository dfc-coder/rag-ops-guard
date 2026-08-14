from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_ops_guard.app import embeddings, object_store, reranker, vector_store
from rag_ops_guard.config import get_settings
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.retrieval.calibration import corpus_fingerprint, derive_separating_threshold
from rag_ops_guard.retrieval.hybrid import KnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver

DATASET_PATH = Path("evaluation/datasets/retrieval-calibration-v1.json")
ARTIFACT_PATH = Path(".local/reranker-calibration.json")


def _dataset_fingerprint() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def _cached(settings_model: str, corpus_hash: str, dataset_hash: str) -> bool:
    if not ARTIFACT_PATH.exists():
        return False
    try:
        payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("reranker_model") == settings_model
        and payload.get("corpus_fingerprint") == corpus_hash
        and payload.get("dataset_fingerprint") == dataset_hash
        and isinstance(payload.get("threshold"), int | float)
    )


def main() -> None:
    settings = get_settings()
    corpus_hash = corpus_fingerprint()
    dataset_hash = _dataset_fingerprint()
    if _cached(settings.reranker_model, corpus_hash, dataset_hash):
        payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        print(f"reranker calibration: cached threshold={float(payload['threshold']):.6f}")
        return

    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    search = KnowledgeSearch(
        embeddings=embeddings(),
        vectors=vector_store(),
        objects=object_store(),
        resolver=EvidenceResolver(),
        reranker=reranker(),
        candidate_k=settings.retrieval_top_k,
        context_k=settings.retrieval_context_k,
        min_reranker_score=0.0,
    )

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    records: list[dict[str, object]] = []

    for sample in samples:
        query = str(sample["query"])
        label = str(sample["label"])
        expected_titles = {str(title) for title in sample.get("expected_titles", [])}
        result = search.search(query, QueryContext(), query_mode="probe")
        admitted_titles = [item.chunk.title for item in result.admitted]

        if label == "positive":
            matches = [item for item in result.admitted if item.chunk.title in expected_titles]
            if not matches:
                raise RuntimeError(
                    f"positive calibration case {sample['id']} did not retrieve an expected title; "
                    f"got={admitted_titles} expected={sorted(expected_titles)}"
                )
            score = max(result.reranker_scores[item.chunk.id] for item in matches)
            positive_scores.append(score)
        elif label == "negative":
            score = result.relevance
            negative_scores.append(score)
        else:
            raise RuntimeError(f"unknown calibration label: {label}")

        records.append(
            {
                "id": sample["id"],
                "label": label,
                "query": query,
                "score": round(score, 6),
                "top_score": result.relevance,
                "admitted_titles": admitted_titles,
            }
        )
        print(
            f"CAL {label:8s} {sample['id']}: score={score:.6f} "
            f"top={result.relevance:.6f} titles={admitted_titles}"
        )

    calibrated = derive_separating_threshold(positive_scores, negative_scores)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "reranker_model": settings.reranker_model,
        "corpus_fingerprint": corpus_hash,
        "dataset_fingerprint": dataset_hash,
        "threshold": calibrated.threshold,
        "min_positive": calibrated.min_positive,
        "max_negative": calibrated.max_negative,
        "separation": calibrated.separation,
        "cases": records,
    }
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "reranker calibration: PASS "
        f"threshold={calibrated.threshold:.6f} "
        f"max_negative={calibrated.max_negative:.6f} "
        f"min_positive={calibrated.min_positive:.6f} "
        f"gap={calibrated.separation:.6f}"
    )


if __name__ == "__main__":
    main()
