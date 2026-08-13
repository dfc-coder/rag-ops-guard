from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

from rag_ops_guard.domain.models import GroundedAnswer, QueryAnalysis


class _QueryAnalysisOutput(QueryAnalysis):
    safety_blocked_message: str = Field(min_length=1, max_length=300)
    insufficient_evidence_message: str = Field(min_length=1, max_length=300)


class LlamaCppChatAdapter:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
        repeat_penalty: float = 1.0,
    ) -> None:
        base = ChatOpenAI(
            base_url=base_url,
            api_key=SecretStr("local"),
            model=model,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_completion_tokens=max_tokens,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
            },
        )
        self._analysis = base.with_structured_output(_QueryAnalysisOutput, method="json_schema")
        self._answer = base.with_structured_output(GroundedAnswer, method="json_schema")

    def analyze_query(self, prompt: str) -> QueryAnalysis:
        result = self._analysis.invoke(prompt)
        return QueryAnalysis.model_validate(result)

    def generate_answer(self, prompt: str) -> GroundedAnswer:
        result = self._answer.invoke(prompt)
        return GroundedAnswer.model_validate(result)
