"""Fase 0 — endurecimiento de configuración sin DB.

Cierra: C1 (extra=ignore), C2 (secretos en claro), C3 (sin validación cruzada),
C6/A10 (URLs sin allowlist), C11 (shadow config RAGAS_*), C12 (secreto fuera de
Settings), INV-1 (floor anulado por max()), INV-2 (coherencia de entorno).
"""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_LOCAL_SERVICE_HOSTS = {"floci", "llama-gen", "llama-embed", "llama-rerank", "ovms-rag"}


def _host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is_local_host(url: str) -> bool:
    host = _host_of(url)
    return host in _LOOPBACK or host in _LOCAL_SERVICE_HOSTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="forbid", env_ignore_empty=True)

    app_env: Literal["local", "local-observed", "ci", "aws"] = "local"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = "test"
    aws_secret_access_key: SecretStr = SecretStr("test")
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
    llm_model: str = "qwen3.5-2b-unsloth-ud-q4-k-xl"
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

    floci_host_port: int = Field(default=4566, ge=1, le=65535)
    llama_gen_host_port: int = Field(default=8080, ge=1, le=65535)
    llama_embed_host_port: int = Field(default=8081, ge=1, le=65535)
    llama_rerank_host_port: int = Field(default=8082, ge=1, le=65535)
    llama_ctx_size: int = Field(default=16384, ge=512, le=131072)
    llama_parallel: int = Field(default=1, ge=1, le=16)

    ragas_judge_provider: Literal["local", "openai", "anthropic"] = "local"
    ragas_judge_model: str = ""
    ragas_judge_base_url: str = ""
    ragas_judge_api_key: SecretStr | None = None
    ragas_judge_timeout_seconds: float = Field(default=120.0, ge=5.0, le=900.0)
    ragas_operation_timeout_seconds: float = Field(default=120.0, ge=5.0, le=900.0)
    ragas_max_retries: int = Field(default=2, ge=0, le=10)
    ragas_max_wait_seconds: float = Field(default=5.0, ge=0.0, le=120.0)
    ragas_max_workers: int = Field(default=2, ge=1, le=32)
    ragas_suite_timeout_seconds: float = Field(default=1800.0, ge=60.0, le=14400.0)

    langsmith_tracing: bool = False
    langsmith_project: str = "rag-ops-guard-local"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: SecretStr | None = None
    langsmith_workspace_id: str | None = None
    rag_debug: bool = False

    @model_validator(mode="after")
    def _check_cross_field_invariants(self) -> "Settings":
        if self.embedding_dimension != self.vector_dimension:
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) debe ser igual a "
                f"vector_dimension ({self.vector_dimension}); divergen -> corrupción "
                f"silenciosa del índice vectorial"
            )
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) debe ser menor que "
                f"chunk_tokens ({self.chunk_tokens})"
            )
        if self.retrieval_context_k > self.retrieval_top_k:
            raise ValueError(
                f"retrieval_context_k ({self.retrieval_context_k}) no puede superar "
                f"retrieval_top_k ({self.retrieval_top_k})"
            )
        if self.llm_analysis_max_tokens >= self.llm_answer_max_tokens:
            raise ValueError("llm_analysis_max_tokens debe ser menor que llm_answer_max_tokens")
        if self.retrieval_min_relevance < self.retrieval_domain_min_relevance:
            raise ValueError(
                f"retrieval_min_relevance ({self.retrieval_min_relevance}) < "
                f"retrieval_domain_min_relevance ({self.retrieval_domain_min_relevance}): "
                "hybrid.py aplica max() sobre ambos, así que este valor sería un "
                "no-op silencioso. El floor de admisión debe ser >= al floor de dominio."
            )
        return self

    @model_validator(mode="after")
    def _check_environment_coherence(self) -> "Settings":
        endpoint_is_local = _is_local_host(self.aws_endpoint_url)
        if self.app_env in ("local", "local-observed", "ci"):
            if not endpoint_is_local:
                raise ValueError(
                    f"app_env={self.app_env} exige aws_endpoint_url local; "
                    f"apunta a {_host_of(self.aws_endpoint_url)!r}"
                )
        elif self.app_env == "aws":
            if endpoint_is_local:
                raise ValueError(
                    "app_env=aws no puede usar un aws_endpoint_url local; "
                    "dejalo vacío para usar el endpoint real de AWS"
                )
            if self.aws_endpoint_url and not self.aws_endpoint_url.startswith("https://"):
                raise ValueError("app_env=aws exige https en aws_endpoint_url")
        return self

    @model_validator(mode="after")
    def _check_model_url_allowlist(self) -> "Settings":
        urls = {
            "llm_base_url": self.llm_base_url,
            "embedding_base_url": self.embedding_base_url,
            "reranker_base_url": self.reranker_base_url,
        }
        for name, url in urls.items():
            scheme = urlparse(url).scheme
            if scheme not in ("http", "https"):
                raise ValueError(f"{name}: esquema {scheme!r} no permitido")
            if self.app_env in ("local", "local-observed", "ci"):
                if not _is_local_host(url):
                    raise ValueError(
                        f"{name} apunta a {_host_of(url)!r}, fuera del host local; "
                        f"prohibido con app_env={self.app_env}"
                    )
            elif self.app_env == "aws" and scheme != "https":
                raise ValueError(f"{name} exige https con app_env=aws")
        return self

    @model_validator(mode="after")
    def _check_ragas_judge(self) -> "Settings":
        if self.ragas_judge_provider != "local":
            if not self.ragas_judge_base_url:
                raise ValueError(
                    "ragas_judge_base_url es obligatorio con "
                    f"ragas_judge_provider={self.ragas_judge_provider}"
                )
            if self.ragas_judge_api_key is None:
                raise ValueError(
                    "ragas_judge_api_key es obligatorio con "
                    f"ragas_judge_provider={self.ragas_judge_provider}"
                )
        return self

    @property
    def resolved_ragas_judge_model(self) -> str:
        return self.ragas_judge_model or self.llm_model

    @property
    def resolved_ragas_judge_base_url(self) -> str:
        return self.ragas_judge_base_url or self.llm_base_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
