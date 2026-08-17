from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "local-observed", "ci", "aws"] = "local"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = "test"
    aws_secret_access_key: str = "test"
    aws_endpoint_url: str = "http://localhost:4566"

    s3_document_bucket: str = "rag-ops-guard-docs-local"
    s3_vector_bucket: str = "rag-ops-guard-vectors-local"
    s3_vector_index: str = "ops-knowledge-v1"

    vector_dimension: int = Field(default=1024, ge=1, le=4096)
    vector_distance_metric: Literal["cosine", "euclidean"] = "cosine"
    retrieval_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_context_k: int = Field(default=4, ge=1, le=20)
    retrieval_domain_min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)

    chunk_tokens: int = Field(default=400, ge=50, le=4000)
    chunk_overlap: int = Field(default=60, ge=0, le=1000)

    llm_base_url: str = "http://localhost:8080/v1"
    llm_model: str = "qwen3.5-0.8b-unsloth-ud-q4-k-xl"
    llm_analysis_max_tokens: int = Field(default=128, ge=32, le=512)
    llm_answer_max_tokens: int = Field(default=512, ge=64, le=1024)
    llm_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0)
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_top_p: float = Field(default=0.8, ge=0, le=1)
    llm_top_k: int = Field(default=20, ge=0, le=100)
    llm_min_p: float = Field(default=0.0, ge=0, le=1)
    llm_presence_penalty: float = Field(default=1.5, ge=-2, le=2)
    llm_repeat_penalty: float = Field(default=1.0, ge=0, le=2)

    embedding_base_url: str = "http://localhost:8081/v1"
    embedding_model: str = "qwen3-embedding-0.6b"
    embedding_dimension: int = Field(default=1024, ge=1, le=4096)
    embedding_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0)

    reranker_base_url: str = "http://localhost:8082"
    reranker_model: str = "qwen3-reranker-0.6b"
    reranker_timeout_seconds: float = Field(default=90.0, ge=5.0, le=300.0)

    langsmith_tracing: bool = False
    langsmith_project: str = "rag-ops-guard-local"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str | None = None
    langsmith_workspace_id: str | None = None
    rag_debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
