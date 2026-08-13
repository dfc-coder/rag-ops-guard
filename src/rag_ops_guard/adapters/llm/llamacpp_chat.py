from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

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
        self._analysis = analysis_model.with_structured_output(
            _QueryAnalysisOutput,
            method="json_schema",
        )
        self._answer = answer_model.with_structured_output(GroundedAnswer, method="json_schema")

    def analyze_query(self, prompt: str) -> QueryAnalysis:
        result = self._analysis.invoke(prompt)
        return QueryAnalysis.model_validate(result)

    def generate_answer(self, prompt: str) -> GroundedAnswer:
        result = self._answer.invoke(prompt)
        return GroundedAnswer.model_validate(result)
