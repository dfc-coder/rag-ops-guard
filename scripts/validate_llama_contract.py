from __future__ import annotations

import json
import os
from typing import Any

import httpx

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-2b-unsloth-ud-q4-k-xl")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

SEARCH_TOOL = {
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

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_structured_response",
        "description": "Submit the final structured response.",
        "parameters": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["text", "citation_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
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


def _tool_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []
    result: list[tuple[str, dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
        else:
            name = str(item.get("name") or "")
            arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"tool arguments were not JSON: {arguments!r}") from exc
            arguments = parsed
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            result.append((name, arguments))
    return result


def validate_forced_structured_submission() -> None:
    response = _post(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Submit a final structured response using the required tool.",
                },
                {"role": "user", "content": "Say status ok."},
            ],
            "tools": [SUBMIT_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "submit_structured_response"},
            },
            "parallel_tool_calls": False,
            "temperature": 0,
        }
    )
    calls = _tool_calls(_message(response))
    selected = [args for name, args in calls if name == "submit_structured_response"]
    if len(selected) != 1:
        raise RuntimeError(f"forced structured submission failed: {_message(response)!r}")
    segments = selected[0].get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError(f"structured submission arguments invalid: {selected[0]!r}")
    print("llama structured submission tool: ready")


def validate_forced_search_protocol() -> None:
    response = _post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Search the internal retry policy."}],
            "tools": [SEARCH_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "search_documents"}},
            "parallel_tool_calls": False,
            "temperature": 0,
        }
    )
    calls = _tool_calls(_message(response))
    if not any(name == "search_documents" for name, _ in calls):
        raise RuntimeError(f"forced tool-call protocol failed: {_message(response)!r}")
    print("llama forced search tool calling: ready")


def validate_automatic_tool_selection() -> None:
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
            "tools": [SEARCH_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0,
        }
    )
    message = _message(response)
    calls = _tool_calls(message)
    if not any(name == "search_documents" for name, _ in calls):
        raise RuntimeError(f"automatic tool selection failed: {message!r}")
    print("llama automatic search tool selection: ready")


def main() -> None:
    validate_forced_search_protocol()
    validate_forced_structured_submission()
    validate_automatic_tool_selection()
    print("LLAMA FUNCTION-CALLING CONTRACT READY")


if __name__ == "__main__":
    main()
