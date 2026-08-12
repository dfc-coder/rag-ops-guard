from __future__ import annotations

import httpx


class LlamaCppTokenCounter:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._url = f"{base_url.removesuffix('/v1')}/tokenize"
        self._timeout = timeout_seconds

    def __call__(self, text: str) -> int:
        response = httpx.post(
            self._url,
            json={"content": text, "add_special": False, "with_pieces": False},
            timeout=self._timeout,
        )
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("llama.cpp tokenize response did not include tokens")
        return len(tokens)
