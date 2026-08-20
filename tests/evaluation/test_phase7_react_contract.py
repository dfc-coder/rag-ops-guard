from __future__ import annotations

from collections import deque
from typing import Any, TypeVar

from rag_ops_guard.agent.conversation import ConversationAgent, ConversationResponse
from rag_ops_guard.domain.models import QueryContext
from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool, ToolCall, ToolResult

T = TypeVar("T")


class EvaluationModel:
    def __init__(self, turns: list[ModelTurn], finals: list[str]) -> None:
        self._turns = deque(turns)
        self._finals = deque(finals)
        self.prompts: list[list[ModelMessage]] = []

    def bind_tools(self, _tools: list[Tool]) -> EvaluationModel:
        return self

    def invoke(self, messages: list[ModelMessage]) -> ModelTurn:
        self.prompts.append(list(messages))
        return self._turns.popleft()

    def invoke_structured(self, _messages: list[ModelMessage], schema: type[T]) -> T:
        return schema.model_validate(  # type: ignore[attr-defined,no-any-return]
            {"segments": [{"text": self._finals.popleft(), "citation_ids": []}]}
        )


class EvaluationTool:
    def __init__(
        self,
        *,
        name: str,
        required_argument: str,
        payload: dict[str, Any],
        trace: list[str],
    ) -> None:
        self.name = name
        self.description = f"Evaluation tool {name}."
        self._required_argument = required_argument
        self._payload = payload
        self._trace = trace

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {self._required_argument: {"type": "string"}},
            "required": [self._required_argument],
            "additionalProperties": False,
        }

    def invoke(self, _arguments: dict[str, Any]) -> ToolResult:
        self._trace.append(self.name)
        return ToolResult(ok=True, payload=self._payload)


def _invoke(agent: ConversationAgent, prompt: str, thread_id: str) -> ConversationResponse:
    result = agent.invoke(prompt, thread_id=thread_id, context=QueryContext())
    assert isinstance(result, ConversationResponse)
    return result


def test_phase7_multitool_evaluation_contract() -> None:
    trace: list[str] = []
    lookup = EvaluationTool(
        name="lookup_rate",
        required_argument="subject",
        payload={"rate": 0.07},
        trace=trace,
    )
    calculator = EvaluationTool(
        name="calculator",
        required_argument="expression",
        payload={"value": 59.5},
        trace=trace,
    )
    model = EvaluationModel(
        turns=[
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="lookup",
                        name="lookup_rate",
                        arguments={"subject": "retry rate"},
                    ),
                )
            ),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="calculate",
                        name="calculator",
                        arguments={"expression": "850 * 0.07"},
                    ),
                )
            ),
            ModelTurn(content="About 60 operations."),
        ],
        finals=["About 60 operations."],
    )
    agent = ConversationAgent(model=model, tools=[lookup, calculator])

    response = _invoke(agent, "Apply the retry rate to 850 operations", "eval-multitool")

    assert trace == ["lookup_rate", "calculator"]
    assert response.tool_calls == 2
    assert response.answer == "About 60 operations."
    assert len(model.prompts) == 3


def test_phase7_multiturn_evaluation_contract() -> None:
    trace: list[str] = []
    calculator = EvaluationTool(
        name="calculator",
        required_argument="expression",
        payload={"value": 60},
        trace=trace,
    )
    model = EvaluationModel(
        turns=[
            ModelTurn(content="30"),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        id="double",
                        name="calculator",
                        arguments={"expression": "30 * 2"},
                    ),
                )
            ),
            ModelTurn(content="60"),
        ],
        finals=["30", "60"],
    )
    agent = ConversationAgent(model=model, tools=[calculator])

    first = _invoke(agent, "What is 10 + 20?", "eval-multiturn")
    second = _invoke(agent, "And double that?", "eval-multiturn")

    assert first.answer == "30"
    assert second.answer == "60"
    assert trace == ["calculator"]
    second_turn_prompt = model.prompts[1]
    assert any(
        message.role == "assistant" and message.content == "30"
        for message in second_turn_prompt
    )
    assert any(
        message.role == "user" and message.content == "And double that?"
        for message in second_turn_prompt
    )
