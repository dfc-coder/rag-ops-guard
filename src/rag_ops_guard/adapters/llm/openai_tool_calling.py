from __future__ import annotations

import copy
import json
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
        Citation IDs are additionally constrained to chunk IDs returned by search_documents,
        so the model cannot invent an ID that the application must reject afterwards.

        Final structured generation is deterministic and uses the effective configured
        completion budget. The same versioned setting therefore governs normal and
        schema-constrained generation without a hidden lower cap.
        """
        validator = getattr(schema, "model_validate_json", None)
        schema_factory = getattr(schema, "model_json_schema", None)
        if not callable(validator) or not callable(schema_factory):
            raise TypeError(
                "structured schema must provide model_json_schema and model_validate_json"
            )

        base_schema = cast(dict[str, Any], schema_factory())
        json_schema = _constrain_citation_ids(base_schema, messages)
        runnable = self._model.bind(
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
            temperature=0.0,
            presence_penalty=0.0,
            max_completion_tokens=self._config.max_completion_tokens,
        )
        response = runnable.invoke(_to_langchain_messages(messages))
        if not isinstance(response, AIMessage):
            raise TypeError(f"expected AIMessage, got {type(response).__name__}")

        content = _content_text(response.content).strip()
        if not content:
            raise ValueError("model returned an empty structured response")
        return cast(T, validator(content))


def _retrieved_citation_ids(messages: list[ModelMessage]) -> list[str]:
    ids: set[str] = set()
    for message in messages:
        if message.role != "tool" or message.name != "search_documents":
            continue
        try:
            result = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        payload = result.get("payload")
        if not isinstance(payload, dict):
            continue
        sources = payload.get("sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            chunk_id = source.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id.strip():
                ids.add(chunk_id.strip())
    return sorted(ids)


def _constrain_citation_ids(
    schema: dict[str, Any],
    messages: list[ModelMessage],
) -> dict[str, Any]:
    constrained = copy.deepcopy(schema)
    allowed_ids = _retrieved_citation_ids(messages)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                citation_ids = properties.get("citation_ids")
                if isinstance(citation_ids, dict):
                    if allowed_ids:
                        citation_ids["items"] = {
                            "type": "string",
                            "enum": allowed_ids,
                        }
                    else:
                        citation_ids["maxItems"] = 0
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(constrained)
    return constrained


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
            continue
        if message.role == "user":
            converted.append(HumanMessage(content=message.content))
            continue
        if message.role == "assistant":
            tool_calls = [
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.arguments,
                    "type": "tool_call",
                }
                for call in message.tool_calls
            ]
            converted.append(AIMessage(content=message.content, tool_calls=tool_calls))
            continue
        if message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("tool message requires tool_call_id")
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
            continue
        raise ValueError(f"unsupported model message role: {message.role}")
    return converted


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
