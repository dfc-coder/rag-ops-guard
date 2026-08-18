from rag_ops_guard.adapters.llm.openai_tool_calling import _structured_input_messages
from rag_ops_guard.ports.interfaces import ModelMessage, ToolCall


def test_structured_input_drops_trailing_plain_assistant_draft() -> None:
    messages = [
        ModelMessage(role="system", content="system"),
        ModelMessage(role="user", content="question"),
        ModelMessage(role="assistant", content="unconstrained draft"),
    ]

    result = _structured_input_messages(messages)

    assert [message.role for message in result] == ["system", "user"]


def test_structured_input_preserves_completed_tool_protocol() -> None:
    messages = [
        ModelMessage(role="user", content="question"),
        ModelMessage(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall(id="call-1", name="search_documents", arguments={"query": "question"}),
            ),
        ),
        ModelMessage(
            role="tool",
            name="search_documents",
            tool_call_id="call-1",
            content='{"ok":true}',
        ),
    ]

    result = _structured_input_messages(messages)

    assert result == messages
