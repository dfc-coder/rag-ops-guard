from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx


def _timings(payload: dict[str, Any]) -> tuple[int, int]:
    timings = payload.get("timings")
    if not isinstance(timings, dict):
        raise SystemExit("llama.cpp response does not expose timings; prompt cache cannot be verified")
    cache_n = int(timings.get("cache_n", 0))
    prompt_n = int(timings.get("prompt_n", 0))
    return cache_n, prompt_n


def _request(base_url: str, prefix: str, question: str) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "qwen3-4b-rag",
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
            "max_tokens": 1,
            "cache_prompt": True,
        },
        timeout=180,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


def main() -> None:
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    nonce = uuid.uuid4().hex
    common = (
        "RAG Ops Guard prompt-cache validation. "
        f"Validation nonce: {nonce}. "
        "This prefix must remain identical across both requests. "
    )
    prefix = common + ("Ground every answer in admitted operational evidence. " * 220)

    first = _request(base_url, prefix, "Return the word alpha.")
    second = _request(base_url, prefix, "Return the word beta.")
    first_cache, first_prompt = _timings(first)
    second_cache, second_prompt = _timings(second)
    first_total = first_cache + first_prompt

    reused = second_cache >= max(64, int(first_total * 0.70))
    reduced_prefill = second_prompt < max(1, int(first_prompt * 0.50))
    passed = reused and reduced_prefill

    result = {
        "passed": passed,
        "first": {"cache_n": first_cache, "prompt_n": first_prompt},
        "second": {"cache_n": second_cache, "prompt_n": second_prompt},
        "cache_reuse_ratio": round(second_cache / max(1, first_total), 4),
    }
    output = Path("artifacts/cache-validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    if not passed:
        raise SystemExit("KV prompt-cache validation failed")


if __name__ == "__main__":
    main()
