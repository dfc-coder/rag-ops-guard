from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_ops_guard.domain.models import GroundedAnswer, QueryAnalysis


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
        common = {
            "base_url": base_url,
            "api_key": SecretStr("local"),
            "model": model,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "timeout": timeout_seconds,
            "max_retries": 0,
            "extra_body": {
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
            },
        }
        analysis_model = ChatOpenAI(
            **common,
            temperature=0.0,
            max_completion_tokens=analysis_max_tokens,
        )
        answer_model = ChatOpenAI(
            **common,
            temperature=temperature,
            max_completion_tokens=answer_max_tokens,
        )
        self._analysis = analysis_model.with_structured_output(QueryAnalysis, method="json_schema")
        self._answer = answer_model.with_structured_output(GroundedAnswer, method="json_schema")

    def analyze_query(self, prompt: str) -> QueryAnalysis:
        result = self._analysis.invoke(prompt)
        return QueryAnalysis.model_validate(result)

    def generate_answer(self, prompt: str) -> GroundedAnswer:
        result = self._answer.invoke(prompt)
        return GroundedAnswer.model_validate(result)
