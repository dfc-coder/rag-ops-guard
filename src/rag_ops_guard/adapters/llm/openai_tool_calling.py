from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool, ToolCall

T = TypeVar("T")


@dataclass(frozen=True)
class _AdapterConfig:
    base_url: str
    model: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repeat_penalty: float
    max_completion_tokens: int
    timeout_seconds: float


def _base_extra_body(config: _AdapterConfig) -> dict[str, Any]:
    return {
        "top_k": config.top_k,
        "min_p": config.min_p,
        "repeat_penalty": config.repeat_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class OpenAIToolCallingAdapter:
    """OpenAI-style tool-calling adapter for the local llama.cpp generation server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repeat_penalty: float,
        max_completion_tokens: int,
        timeout_seconds: float,
        runnable: Any | None = None,
    ) -> None:
        self._config = _AdapterConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            max_completion_tokens=max_completion_tokens,
            timeout_seconds=timeout_seconds,
        )
        self._model = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout_seconds,
            max_retries=0,
            extra_body=_base_extra_body(self._config),
        )
        self._runnable = runnable or self._model

    def bind_tools(self, tools: list[Tool]) -> OpenAIToolCallingAdapter:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema(),
                },
            }
            for tool in tools
        ]
        runnable = self._model.bind_tools(
            definitions,
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        config = self._config
        return OpenAIToolCallingAdapter(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            presence_penalty=config.presence_penalty,
            repeat_penalty=config.repeat_penalty,
            max_completion_tokens=config.max_completion_tokens,
            timeout_seconds=config.timeout_seconds,
            runnable=runnable,
        )

    def invoke(self, messages: list[ModelMessage]) -> ModelTurn:
        response = self._runnable.invoke(_to_langchain_messages(messages))
        if not isinstance(response, AIMessage):
            raise TypeError(f"expected AIMessage, got {type(response).__name__}")
        calls = tuple(_tool_call(call) for call in response.tool_calls)
        metadata = (
            response.response_metadata if isinstance(response.response_metadata, dict) else {}
        )
        finish_reason = metadata.get("finish_reason")
        return ModelTurn(
            content=_content_text(response.content),
            tool_calls=calls,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )

    def invoke_structured(self, messages: list[ModelMessage], schema: type[T]) -> T:
        """Generate schema-constrained JSON and validate it with Pydantic.

        ReAct still uses native tool calling. The final response intentionally uses
        llama.cpp JSON-schema constrained generation instead of a forced function call,
        because some local chat templates ignore forced tool_choice for that last turn.
        """
        validator = getattr(schema, "model_validate_json", None)
        schema_factory = getattr(schema, "model_json_schema", None)
        if not callable(validator) or not callable(schema_factory):
            raise TypeError(
                "structured schema must provide model_json_schema and model_validate_json"
            )

        json_schema = cast(dict[str, Any], schema_factory())
        runnable = self._model.bind(
            response_format={
                "type": "json_object",
                "schema": json_schema,
            }
        )
        response = runnable.invoke(_to_langchain_messages(messages))
        if not isinstance(response, AIMessage):
            raise TypeError(f"expected AIMessage, got {type(response).__name__}")

        content = _content_text(response.content).strip()
        if not content:
            raise ValueError("model returned an empty structured response")
        return cast(T, validator(content))


def _tool_call(call: Any) -> ToolCall:
    return ToolCall(
        id=str(call.get("id") or ""),
        name=str(call.get("name") or ""),
        arguments=dict(call.get("args") or {}),
    )


def _to_langchain_messages(messages: list[ModelMessage]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=[
                        {
                            "id": call.id,
                            "name": call.name,
                            "args": call.arguments,
                            "type": "tool_call",
                        }
                        for call in message.tool_calls
                    ],
                )
            )
        elif message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported message role: {message.role}")
    return converted


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
    )
