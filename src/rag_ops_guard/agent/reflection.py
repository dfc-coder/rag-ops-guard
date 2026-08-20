from __future__ import annotations

import json
from typing import Protocol

from rag_ops_guard.ports.interfaces import ModelMessage, Tool, ToolCall, ToolCallingModel, ToolResult

REFLECTION_PROMPT = """
You repair failed tool-use attempts for another assistant.

Return one short correction for the next attempt.
Do not answer the user's request.
Do not invent tool results.
Prefer correcting the tool name or arguments using the available tool schemas.
If no available tool can resolve the failure, say that the assistant should stop retrying.
""".strip()


class Reflector(Protocol):
    def reflect(
        self,
        *,
        goal: str,
        call: ToolCall,
        result: ToolResult,
        tools: tuple[Tool, ...],
    ) -> str: ...


class ModelReflector:
    """Produce one bounded correction using the unbound base model."""

    def __init__(self, model: ToolCallingModel) -> None:
        self._model = model

    def reflect(
        self,
        *,
        goal: str,
        call: ToolCall,
        result: ToolResult,
        tools: tuple[Tool, ...],
    ) -> str:
        tool_descriptors = [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema(),
            }
            for tool in tools
        ]
        payload = {
            "goal": goal,
            "failed_call": {
                "name": call.name,
                "arguments": call.arguments,
            },
            "failure": result.model_dump(mode="json"),
            "available_tools": tool_descriptors,
        }
        turn = self._model.invoke(
            [
                ModelMessage(role="system", content=REFLECTION_PROMPT),
                ModelMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False),
                ),
            ]
        )
        correction = turn.content.strip()
        if correction:
            return correction
        return (
            f"The previous tool call failed with {result.reason or 'tool_failed'}. "
            "Correct the tool name or arguments before retrying."
        )
