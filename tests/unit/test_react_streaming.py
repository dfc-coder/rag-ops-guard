from __future__ import annotations

from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext


class SuccessGraph:
    def __init__(self, answer: str = "Hola mundo") -> None:
        self.answer = answer
        self.inputs: list[dict[str, Any]] = []

    def stream(self, inputs: dict[str, Any], **_: Any):
        self.inputs.append(inputs)
        midpoint = max(1, len(self.answer) // 2)
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content=self.answer[:midpoint]),
                {"langgraph_node": "agent"},
            ),
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content=self.answer[midpoint:]),
                {"langgraph_node": "agent"},
            ),
        }
        final_messages = [*inputs["messages"], AIMessage(content=self.answer)]
        yield {"type": "values", "data": {"messages": final_messages}}


class TruncatedGraph:
    def stream(self, inputs: dict[str, Any], **_: Any):
        partial = "```perl\nsub encode {\n"
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content=partial), {"langgraph_node": "agent"}),
        }
        final_messages = [
            *inputs["messages"],
            AIMessage(content=partial, response_metadata={"finish_reason": "length"}),
        ]
        yield {"type": "values", "data": {"messages": final_messages}}


class APITimeoutError(Exception):
    pass


class TimeoutGraph:
    def stream(self, _inputs: dict[str, Any], **_: Any):
        yield {
            "type": "messages",
            "data": (AIMessageChunk(content="respuesta parcial"), {"langgraph_node": "agent"}),
        }
        raise APITimeoutError("model stalled")


def make_agent(graph: Any) -> ReactAgent:
    agent = object.__new__(ReactAgent)
    agent._history_guard = Lock()
    agent._histories = {}
    agent._thread_locks = {}
    agent._agent = graph
    return agent


def text_content(messages: list[BaseMessage]) -> list[str]:
    return [str(message.content) for message in messages]


def test_stream_yields_immediate_status_tokens_and_terminal_response() -> None:
    graph = SuccessGraph("Hola mundo")
    agent = make_agent(graph)

    events = list(
        agent.stream(
            "Hola",
            thread_id="thread-1",
            context=QueryContext(),
        )
    )

    assert events[0].kind == "status"
    token_events = [event for event in events if event.kind == "token"]
    assert len(token_events) == 2
    assert token_events[-1].text == "Hola mundo"
    assert events[-1].kind == "done"
    assert events[-1].text == "Hola mundo"
    assert text_content(agent._histories["thread-1"]) == ["Hola", "Hola mundo"]


def test_timeout_is_friendly_and_does_not_commit_failed_turn() -> None:
    agent = make_agent(TimeoutGraph())
    previous = [HumanMessage(content="Antes"), AIMessage(content="Respuesta anterior")]
    agent._histories["thread-1"] = list(previous)

    events = list(
        agent.stream(
            "Este turno falla",
            thread_id="thread-1",
            context=QueryContext(),
        )
    )

    terminal = events[-1]
    assert terminal.kind == "error"
    assert "respuesta parcial" in terminal.text
    assert "conversación anterior sigue intacta" in terminal.text
    assert text_content(agent._histories["thread-1"]) == text_content(previous)


def test_truncated_turn_is_recoverable_and_not_committed() -> None:
    agent = make_agent(TruncatedGraph())
    previous = [HumanMessage(content="Antes"), AIMessage(content="Respuesta anterior")]
    agent._histories["thread-1"] = list(previous)

    events = list(
        agent.stream(
            "Genera código",
            thread_id="thread-1",
            context=QueryContext(),
        )
    )

    terminal = events[-1]
    assert terminal.kind == "error"
    assert terminal.finish_reason == "length"
    assert "alcanzó el límite de generación" in terminal.text
    assert text_content(agent._histories["thread-1"]) == text_content(previous)


def test_next_turn_continues_from_last_successful_history_after_failure() -> None:
    agent = make_agent(TimeoutGraph())
    previous = [HumanMessage(content="Antes"), AIMessage(content="Respuesta anterior")]
    agent._histories["thread-1"] = list(previous)

    list(
        agent.stream(
            "Mensaje fallido",
            thread_id="thread-1",
            context=QueryContext(),
        )
    )

    success = SuccessGraph("Seguimos")
    agent._agent = success
    events = list(
        agent.stream(
            "Continuemos",
            thread_id="thread-1",
            context=QueryContext(),
        )
    )

    assert events[-1].kind == "done"
    sent_messages = success.inputs[0]["messages"]
    assert text_content(sent_messages) == ["Antes", "Respuesta anterior", "Continuemos"]
    assert "Mensaje fallido" not in text_content(sent_messages)
    assert text_content(agent._histories["thread-1"]) == [
        "Antes",
        "Respuesta anterior",
        "Continuemos",
        "Seguimos",
    ]


def test_invoke_reports_recoverable_failure_without_raising() -> None:
    agent = make_agent(TimeoutGraph())

    response = agent.invoke("falla", thread_id="thread-1", context=QueryContext())

    assert response.failed is True
    assert "conversación anterior sigue intacta" in response.answer
