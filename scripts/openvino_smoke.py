from __future__ import annotations

import os
from time import perf_counter

import httpx

BASE_URL = os.environ.get("OPENVINO_BASE_URL", "http://127.0.0.1:8083/v3").rstrip("/")
EMBEDDING_MODEL = os.environ.get(
    "OVMS_EMBEDDING_MODEL",
    "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
)
RERANKER_MODEL = os.environ.get(
    "OVMS_RERANKER_MODEL",
    "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov",
)


def timed_post(path: str, payload: dict[str, object]) -> tuple[float, httpx.Response]:
    started = perf_counter()
    try:
        response = httpx.post(f"{BASE_URL}/{path}", json=payload, timeout=120.0)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"OpenVINO backend is not reachable at {BASE_URL}. "
            "Run `make openvino-models` and then `make openvino-up` before this smoke test."
        ) from exc
    elapsed = perf_counter() - started
    response.raise_for_status()
    return elapsed, response


def main() -> None:
    embedding_seconds, embedding = timed_post(
        "embeddings",
        {
            "model": EMBEDDING_MODEL,
            "input": "How many automated retries does Calypso allow?",
        },
    )
    vector = embedding.json()["data"][0]["embedding"]
    print(f"embedding: {embedding_seconds:.3f}s · dimensions={len(vector)}")

    documents = [
        "Transient Calypso timeouts may be retried automatically a maximum of three times.",
        "After the third automated retry fails, escalate to Treasury Integrations.",
        "SendGrid notification failures use a separate runbook.",
        "The cafeteria opens at nine.",
    ]
    rerank_seconds, rerank = timed_post(
        "rerank",
        {
            "model": RERANKER_MODEL,
            "query": "What happens after the third Calypso retry?",
            "documents": documents,
            "top_n": len(documents),
        },
    )
    results = rerank.json().get("results", [])
    print(f"rerank: {rerank_seconds:.3f}s · documents={len(documents)}")
    for item in results:
        print(
            f"  #{item['index']} score={item['relevance_score']:.4f} · {documents[item['index']]}"
        )


if __name__ == "__main__":
    main()
