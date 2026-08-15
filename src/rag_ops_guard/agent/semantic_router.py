from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from rag_ops_guard.config import get_settings

logger = logging.getLogger(__name__)


class ContextRelation(StrEnum):
    SAME = "same"
    NEW = "new"
    NONE = "none"


class TurnOperation(StrEnum):
    ANSWER = "answer"
    TRANSFORM = "transform"
    CATALOG = "catalog"


class TurnDecision(BaseModel):
    """Semantic routing contract emitted by the local model."""

    requires_grounding: bool = Field(
        description=(
            "True when answering the current request requires private/internal operational facts "
            "or verification against the organization's knowledge base."
        )
    )
    relation_to_context: ContextRelation = Field(
        description=(
            "same when the request depends on the supplied prior conversation/topic; new when it "
            "introduces a different subject; none when it is self-contained and independent."
        )
    )
    operation: TurnOperation = Field(
        description=(
            "answer for a factual/explanatory response, transform for rewriting/summarizing/"
            "translating/reformatting/implementing already supplied content without requesting a "
            "new fact, catalog when asking what internal documentation is available."
        )
    )
    standalone_query: str | None = Field(
        default=None,
        description=(
            "When grounding is required, rewrite the user's request as a self-contained retrieval "
            "query that resolves contextual references using the supplied conversation context. "
            "Otherwise return null."
        ),
    )


@dataclass(frozen=True)
class SemanticConversationContext:
    has_grounded_topic: bool
    grounded_topic: str | None
    last_grounded_query: str | None
    has_active_evidence: bool
    last_user_message: str | None
    last_assistant_message: str | None
    system_filter: str | None
    environment_filter: str | None
    api_version_filter: str | None

    def render(self) -> str:
        return json.dumps(
            {
                "has_grounded_topic": self.has_grounded_topic,
                "grounded_topic": self.grounded_topic,
                "last_grounded_query": self.last_grounded_query,
                "has_active_evidence": self.has_active_evidence,
                "last_user_message": self.last_user_message,
                "last_assistant_message": self.last_assistant_message,
                "system_filter": self.system_filter,
                "environment_filter": self.environment_filter,
                "api_version_filter": self.api_version_filter,
            },
            ensure_ascii=False,
        )


ROUTER_PROMPT = """
You are the semantic routing component of a grounded conversational assistant.
Your only job is to emit the TurnDecision tool call. Do not answer the user.

Interpret meaning, not surface wording. Do not rely on entity allowlists, phrase tables, language-
specific prefixes, capitalization rules, or regular-expression-style matching.

Decision semantics:
- requires_grounding=true only when the requested answer needs private/internal operational facts or
  must be verified against the organization's knowledge base. General knowledge, ordinary chat, and
  self-contained coding do not require grounding.
- relation_to_context=same when the current request semantically depends on the supplied previous
  exchange or grounded topic, even if the subject is omitted or referred to indirectly.
- relation_to_context=new when the user introduces a different subject/entity from the supplied
  context. Use none for an independent self-contained request.
- operation=transform when the request only transforms, reformats, translates, summarizes, explains,
  or implements information already present in context and does not ask for an additional factual
  claim. Use answer when a new factual/explanatory claim is requested.
- operation=catalog only when the user asks what internal documentation/knowledge is available.
- standalone_query is required whenever requires_grounding=true. It must be a concise, self-contained
  retrieval query preserving the actual subject and the current information need. Resolve references
  from context; do not invent facts. Otherwise standalone_query must be null.

If uncertain whether an operational claim needs internal evidence, prefer requires_grounding=true.
""".strip()


class SemanticTurnResolver:
    """Resolve open-ended user language into a validated semantic routing contract."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_settings(cls) -> SemanticTurnResolver:
        settings = get_settings()
        base_model = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=SecretStr("local"),
            model=settings.llm_model,
            temperature=0.0,
            max_completion_tokens=settings.llm_analysis_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
            extra_body={
                "top_k": settings.llm_top_k,
                "min_p": 0.0,
                "repeat_penalty": settings.llm_repeat_penalty,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        bound = base_model.bind_tools(
            [TurnDecision],
            tool_choice=TurnDecision.__name__,
            strict=True,
        )
        return cls(bound)

    def resolve(
        self,
        message: str,
        context: SemanticConversationContext,
    ) -> TurnDecision:
        response = self._model.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=(
                        f"Conversation context:\n{context.render()}\n\n"
                        f"Current user message:\n{message.strip()}"
                    )
                ),
            ]
        )
        if not isinstance(response, AIMessage):
            raise ValueError("semantic router returned a non-AI message")
        if len(response.tool_calls) != 1:
            raise ValueError(
                f"semantic router returned {len(response.tool_calls)} tool calls; expected exactly 1"
            )
        call = response.tool_calls[0]
        if call.get("name") != TurnDecision.__name__:
            raise ValueError(f"semantic router returned unexpected tool {call.get('name')!r}")
        args = call.get("args")
        if not isinstance(args, dict):
            raise ValueError("semantic router tool args are not an object")
        decision = TurnDecision.model_validate(args)
        if decision.requires_grounding and not (decision.standalone_query or "").strip():
            raise ValueError("grounded semantic decision omitted standalone_query")
        if not decision.requires_grounding and decision.standalone_query is not None:
            decision = decision.model_copy(update={"standalone_query": None})
        logger.info(
            "semantic-turn-decision grounding=%s relation=%s operation=%s query=%r",
            decision.requires_grounding,
            decision.relation_to_context.value,
            decision.operation.value,
            decision.standalone_query,
        )
        return decision
