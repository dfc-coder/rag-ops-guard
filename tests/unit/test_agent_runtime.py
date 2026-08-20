from __future__ import annotations

from typing import Any

from rag_ops_guard.agent.context import current_query_context
from rag_ops_guard.agent.runtime import ToolRuntime
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports.interfaces import ToolCall, ToolResult


class ContextTool:
    name = "context_tool"
    description = "Return the request context visible during execution."

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[QueryContext] = []

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }

    def invoke(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(dict(arguments))
        self.contexts.append(current_query_context())
        return ToolResult(ok=True, payload={"value": arguments["value"]})


def test_runtime_rejects_missing_required_argument_before_invocation() -> None:
    tool = ContextTool()
    runtime = ToolRuntime([tool])

    execution = runtime.execute(
        ToolCall(id="bad", name="context_tool", arguments={}),
        QueryContext(),
    )

    assert execution.verification.ok is False
    assert execution.verification.retryable is True
    assert execution.result.reason == "missing_required_argument:value"
    assert tool.calls == []


def test_runtime_rejects_unknown_tool_deterministically() -> None:
    runtime = ToolRuntime([])

    execution = runtime.execute(
        ToolCall(id="unknown", name="missing_tool", arguments={}),
        QueryContext(),
    )

    assert execution.verification.ok is False
    assert execution.verification.retryable is True
    assert execution.result.reason == "unknown_tool"


def test_runtime_binds_query_context_only_during_tool_execution() -> None:
    tool = ContextTool()
    runtime = ToolRuntime([tool])
    context = QueryContext(system="payments", environment="production")

    execution = runtime.execute(
        ToolCall(
            id="context",
            name="context_tool",
            arguments={"value": "ok"},
        ),
        context,
    )

    assert execution.verification.ok is True
    assert tool.calls == [{"value": "ok"}]
    assert tool.contexts == [context]
    assert current_query_context() == QueryContext()
