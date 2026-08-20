from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rag_ops_guard.agent.context import bind_query_context
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports.interfaces import Tool, ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class ToolExecution:
    call: ToolCall
    result: ToolResult
    verification: VerificationResult


class ToolVerifier:
    """Deterministically validate tool calls and tool results."""

    def verify_call(self, call: ToolCall, tool: Tool | None) -> VerificationResult:
        if tool is None:
            return VerificationResult(ok=False, reason="unknown_tool", retryable=True)
        return _verify_arguments(call.arguments, tool.schema())

    def verify_result(self, result: ToolResult) -> VerificationResult:
        if result.ok:
            return VerificationResult(ok=True)
        return VerificationResult(
            ok=False,
            reason=result.reason or "tool_failed",
            retryable=True,
        )


class ToolRuntime:
    """Execute injected tools behind deterministic validation and context binding."""

    def __init__(
        self,
        tools: list[Tool],
        verifier: ToolVerifier | None = None,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._verifier = verifier or ToolVerifier()

    def execute(self, call: ToolCall, context: QueryContext) -> ToolExecution:
        tool = self._tools.get(call.name)
        verification = self._verifier.verify_call(call, tool)
        if not verification.ok:
            return ToolExecution(
                call=call,
                result=ToolResult(
                    ok=False,
                    payload={"tool": call.name, "arguments": call.arguments},
                    reason=verification.reason,
                ),
                verification=verification,
            )

        assert tool is not None
        try:
            with bind_query_context(context):
                result = tool.invoke(call.arguments)
        except Exception:
            logger.exception("tool execution failed tool=%s", call.name)
            result = ToolResult(
                ok=False,
                payload={"tool": call.name},
                reason="tool_exception",
            )

        return ToolExecution(
            call=call,
            result=result,
            verification=self._verifier.verify_result(result),
        )


def _verify_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
) -> VerificationResult:
    schema_type = schema.get("type")
    if schema_type not in {None, "object"}:
        return VerificationResult(
            ok=False,
            reason="unsupported_tool_schema",
            retryable=False,
        )

    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [name for name in required if name not in arguments]
        if missing:
            return VerificationResult(
                ok=False,
                reason=f"missing_required_argument:{','.join(map(str, missing))}",
                retryable=True,
            )

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return VerificationResult(
                ok=False,
                reason=f"unknown_argument:{','.join(unknown)}",
                retryable=True,
            )

    for name, value in arguments.items():
        definition = properties.get(name)
        if not isinstance(definition, dict):
            continue
        expected = definition.get("type")
        if isinstance(expected, list):
            valid = any(_matches_json_type(value, item) for item in expected)
        else:
            valid = _matches_json_type(value, expected)
        if not valid:
            return VerificationResult(
                ok=False,
                reason=f"invalid_argument_type:{name}",
                retryable=True,
            )

    return VerificationResult(ok=True)


def _matches_json_type(value: Any, expected: Any) -> bool:
    if expected is None:
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True
