from __future__ import annotations

import json
import os
from typing import Any

import httpx

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-2b-unsloth-ud-q4-k-xl")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the ingested internal document corpus. For organization-specific policy, "
            "retry, timeout, SLA, ownership, incident, production or runbook facts, call this "
            "tool before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
    if not response.is_success:
        raise RuntimeError(f"llama.cpp contract HTTP {response.status_code}: {response.text[:2000]}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("llama.cpp returned a non-object response")
    return data


def _message(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"llama.cpp response missing message: {payload!r}") from exc
    if not isinstance(message, dict):
        raise RuntimeError(f"llama.cpp message has unexpected shape: {message!r}")
    return message


def _tool_names(message: dict[str, Any]) -> list[str]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
        elif item.get("name"):
            names.append(str(item["name"]))
    return names


def validate_structured_output() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    base = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Return status ok."}],
        "temperature": 0,
    }
    try:
        response = _post(
            {**base, "response_format": {"type": "json_schema", "schema": schema}}
        )
        mode = "json_schema"
    except RuntimeError as exc:
        if "Failed to initialize samplers" not in str(exc):
            raise
        response = _post(
            {
                **base,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only JSON matching this schema: "
                            + json.dumps(schema, separators=(",", ":"))
                        ),
                    },
                    *base["messages"],
                ],
                "response_format": {"type": "json_object"},
            }
        )
        mode = "json_object-fallback"
    content = str(_message(response).get("content") or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"structured output was not JSON: {content!r}") from exc
    if parsed != {"status": "ok"}:
        raise RuntimeError(f"structured output contract mismatch: {parsed!r}")
    print(f"llama structured output: ready ({mode})")


def validate_tool_protocol() -> None:
    forced = _post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Search the internal retry policy."}],
            "tools": [TOOL],
            "tool_choice": {"type": "function", "function": {"name": "search_documents"}},
            "parallel_tool_calls": False,
            "temperature": 0,
        }
    )
    if "search_documents" not in _tool_names(_message(forced)):
        raise RuntimeError(f"forced tool-call protocol failed: {_message(forced)!r}")
    print("llama forced tool calling: ready")


def validate_tool_selection() -> None:
    response = _post(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You have access to an internal document corpus through search_documents. "
                        "For organization-specific facts, always search before answering and never "
                        "claim that you lack access before searching."
                    ),
                },
                {
                    "role": "user",
                    "content": "How many times can a Calypso timeout be retried?",
                },
            ],
            "tools": [TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0,
        }
    )
    message = _message(response)
    if "search_documents" not in _tool_names(message):
        raise RuntimeError(f"automatic tool selection failed: {message!r}")
    print("llama automatic tool selection: ready")


def main() -> None:
    validate_structured_output()
    validate_tool_protocol()
    validate_tool_selection()
    print("LLAMA GENERATION CONTRACT READY")


if __name__ == "__main__":
    main()
