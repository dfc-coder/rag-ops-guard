from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, LengthFinishReasonError
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from rag_ops_guard.domain.models import GroundedAnswer, QueryAnalysis


class _QueryAnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_question: str = Field(min_length=1)
    systems: list[str] = Field(default_factory=list)
    environment: Literal["production", "staging"] | None = None
    api_version: str | None = None
    requires_clarification: bool
    clarification_question: str | None = None
    safety_category: Literal["normal", "secret_extraction", "policy_bypass"]
    fallback_message: str = Field(min_length=1, max_length=180)


class _QueryRewriteOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    standalone_query: str = Field(min_length=1, max_length=2000)


class _GroundedAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


class _GroundedDraftOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)


_REWRITE_SYSTEM_PROMPT = """
Rewrite a conversational follow-up into one standalone retrieval query.
Use only CURRENT_QUESTION, PREVIOUS_GROUNDED_QUERY and SOURCE_TITLES.
Do not answer the question and do not add facts.
Preserve named systems, APIs, identifiers and the language of CURRENT_QUESTION.
If CURRENT_QUESTION is already self-contained, return it unchanged.
Return only the structured schema requested by the caller.
""".strip()


def _trusted_followup_fallback(
    current_question: str,
    previous_query: str,
    source_titles: list[str],
) -> str:
    titles = list(dict.fromkeys(title.strip() for title in source_titles if title.strip()))
    parts = [previous_query.strip(), current_question.strip()]
    if titles:
        parts.append(f"Relevant sources: {' | '.join(titles)}")
    return "\n".join(part for part in parts if part)


class LlamaCppChatAdapter:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        analysis_max_tokens: int,
        answer_max_tokens: int,
        timeout_seconds: float,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repeat_penalty: float = 1.0,
        answer_system_prompt: str = "",
        chat_system_prompt: str = "",
    ) -> None:
        extra_body = {
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
        }
        analysis_model = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
            temperature=0.0,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_completion_tokens=analysis_max_tokens,
            timeout=timeout_seconds,
            max_retries=0,
            extra_body=extra_body,
        )
        answer_model = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_completion_tokens=answer_max_tokens,
            timeout=timeout_seconds,
            max_retries=0,
            extra_body=extra_body,
        )
        self._answer_system_prompt = answer_system_prompt
        self._chat_system_prompt = chat_system_prompt
        # Light chat is intentionally deterministic; grounded answers retain the configured sampler.
        self._chat = analysis_model
        self._analysis = analysis_model.with_structured_output(
            _QueryAnalysisOutput,
            method="json_schema",
        )
        self._rewrite = analysis_model.with_structured_output(
            _QueryRewriteOutput,
            method="json_schema",
        )
        # Legacy workflow output retains status/citation decisions for compatibility.
        self._answer = answer_model.with_structured_output(
            _GroundedAnswerOutput,
            method="json_schema",
        )
        # Conversational-agent output is intentionally answer-only. Retrieval/admission code owns
        # evidence sufficiency and the application owns source attachment.
        self._grounded_draft = answer_model.with_structured_output(
            _GroundedDraftOutput,
            method="json_schema",
        )

    def analyze_query(self, prompt: str) -> QueryAnalysis:
        result = self._analysis.invoke(prompt)
        if isinstance(result, BaseModel):
            return QueryAnalysis.model_validate(result.model_dump())
        return QueryAnalysis.model_validate(result)

    def rewrite_query(
        self,
        current_question: str,
        previous_query: str,
        source_titles: list[str],
    ) -> str:
        payload = {
            "CURRENT_QUESTION": current_question.strip(),
            "PREVIOUS_GROUNDED_QUERY": previous_query.strip(),
            "SOURCE_TITLES": source_titles,
        }
        try:
            result = self._rewrite.invoke(
                [
                    SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            if isinstance(result, BaseModel):
                parsed = _QueryRewriteOutput.model_validate(result.model_dump())
            else:
                parsed = _QueryRewriteOutput.model_validate(result)
            return parsed.standalone_query.strip()
        except (APITimeoutError, LengthFinishReasonError, ValidationError):
            return _trusted_followup_fallback(
                current_question=current_question,
                previous_query=previous_query,
                source_titles=source_titles,
            )

    def generate_chat(self, messages: list[BaseMessage]) -> str:
        request = list(messages)
        if self._chat_system_prompt:
            request.insert(0, SystemMessage(content=self._chat_system_prompt))
        result = self._chat.invoke(request)
        return str(result.content)

    def generate_answer(
        self,
        prompt: str,
        history: list[BaseMessage] | None = None,
    ) -> GroundedAnswer:
        request: list[BaseMessage] = list(history or [])
        if self._answer_system_prompt:
            request.insert(0, SystemMessage(content=self._answer_system_prompt))
        request.append(HumanMessage(content=prompt))
        result = self._answer.invoke(request)
        if isinstance(result, BaseModel):
            return GroundedAnswer.model_validate(result.model_dump())
        return GroundedAnswer.model_validate(result)

    def generate_grounded_text(self, prompt: str) -> str:
        request: list[BaseMessage] = []
        if self._answer_system_prompt:
            request.append(SystemMessage(content=self._answer_system_prompt))
        request.append(HumanMessage(content=prompt))
        result = self._grounded_draft.invoke(request)
        if isinstance(result, BaseModel):
            parsed = _GroundedDraftOutput.model_validate(result.model_dump())
        else:
            parsed = _GroundedDraftOutput.model_validate(result)
        return parsed.answer.strip()
