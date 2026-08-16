from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_ops_guard.ports.interfaces import ModelMessage, ModelTurn, Tool, ToolCall

T = TypeVar("T")


class OpenAIToolCallingAdapter:
    """OpenAI-compatible tool-calling adapter used with the local llama.cpp server."""

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
        self._config = {
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repeat_penalty": repeat_penalty,
            "max_completion_tokens": max_completion_tokens,
            "timeout_seconds": timeout_seconds,
        }
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
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "chat_template_kwargs": {"enable_thinking": False},
            },
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
        runnable = self._model.bind_tools(definitions, parallel_tool_calls=False)
        return OpenAIToolCallingAdapter(runnable=runnable, **self._config)

    def invoke(self, messages: list[ModelMessage]) -> ModelTurn:
        response = self._runnable.invoke(_to_langchain_messages(messages))
        if not isinstance(response, AIMessage):
            raise TypeError(f"expected AIMessage, got {type(response).__name__}")
        calls = tuple(
            ToolCall(
                id=str(call.get("id") or ""),
                name=str(call.get("name") or ""),
                arguments=dict(call.get("args") or {}),
            )
            for call in response.tool_calls
        )
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
        structured = self._model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
        )
        result = structured.invoke(_to_langchain_messages(messages))
        if isinstance(result, schema):
            return result
        validator = getattr(schema, "model_validate", None)
        if callable(validator):
            return cast(T, validator(result))
        return cast(T, result)


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
        else:  # pragma: no cover - ModelMessage constrains the role.
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
