from functools import lru_cache

from rag_ops_guard.adapters.aws.s3_store import S3ObjectStore
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm.openai_tool_calling import OpenAIToolCallingAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter
from rag_ops_guard.agent.catalog import KnowledgeCatalog
from rag_ops_guard.agent.conversation import ConversationAgent
from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.runtime import EffectiveConfig, resolve_effective_config
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.observability.langsmith import configure_langsmith
from rag_ops_guard.retrieval.resilient import ResilientKnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver

_SETTINGS_BY_HASH: dict[str, Settings] = {}
_MAX_CONFIG_GENERATIONS = 8


def _effective() -> EffectiveConfig:
    effective = resolve_effective_config()
    _SETTINGS_BY_HASH[effective.config_hash] = effective.settings
    while len(_SETTINGS_BY_HASH) > _MAX_CONFIG_GENERATIONS:
        oldest = next(iter(_SETTINGS_BY_HASH))
        _SETTINGS_BY_HASH.pop(oldest, None)
    return effective


def _settings(config_hash: str) -> Settings:
    settings = _SETTINGS_BY_HASH.get(config_hash)
    if settings is None:
        effective = _effective()
        if effective.config_hash != config_hash:
            raise RuntimeError("configuration generation changed during dependency construction")
        return effective.settings
    return settings


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _object_store(config_hash: str) -> S3ObjectStore:
    settings = _settings(config_hash)
    return S3ObjectStore(
        bucket=settings.s3_document_bucket,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
    )


def object_store() -> S3ObjectStore:
    effective = _effective()
    return _object_store(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _embeddings(config_hash: str) -> LlamaCppEmbeddingAdapter:
    settings = _settings(config_hash)
    return LlamaCppEmbeddingAdapter(
        settings.embedding_base_url,
        settings.embedding_model,
        settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )


def embeddings() -> LlamaCppEmbeddingAdapter:
    effective = _effective()
    return _embeddings(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _reranker(config_hash: str) -> LlamaCppRerankerAdapter:
    settings = _settings(config_hash)
    return LlamaCppRerankerAdapter(
        settings.reranker_base_url,
        settings.reranker_model,
        settings.reranker_timeout_seconds,
    )


def reranker() -> LlamaCppRerankerAdapter:
    effective = _effective()
    return _reranker(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _vector_store(config_hash: str) -> S3VectorsStore:
    settings = _settings(config_hash)
    return S3VectorsStore(
        vector_bucket=settings.s3_vector_bucket,
        index_name=settings.s3_vector_index,
        object_store=_object_store(config_hash),
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
    )


def vector_store() -> S3VectorsStore:
    effective = _effective()
    return _vector_store(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _conversation_model(config_hash: str) -> OpenAIToolCallingAdapter:
    settings = _settings(config_hash)
    configure_langsmith(settings)
    return OpenAIToolCallingAdapter(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        top_k=settings.llm_top_k,
        min_p=settings.llm_min_p,
        presence_penalty=settings.llm_presence_penalty,
        repeat_penalty=settings.llm_repeat_penalty,
        max_completion_tokens=settings.llm_answer_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def conversation_model() -> OpenAIToolCallingAdapter:
    effective = _effective()
    return _conversation_model(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _ingestion_service(config_hash: str) -> IngestionService:
    settings = _settings(config_hash)
    return IngestionService(
        object_store=_object_store(config_hash),
        vector_store=_vector_store(config_hash),
        embeddings=_embeddings(config_hash),
        chunker=MarkdownChunker(
            token_counter=LlamaCppTokenCounter(settings.llm_base_url),
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
        ),
    )


def ingestion_service() -> IngestionService:
    effective = _effective()
    return _ingestion_service(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _knowledge_search(config_hash: str) -> ResilientKnowledgeSearch:
    settings = _settings(config_hash)
    return ResilientKnowledgeSearch(
        embeddings=_embeddings(config_hash),
        vectors=_vector_store(config_hash),
        objects=_object_store(config_hash),
        resolver=EvidenceResolver(),
        reranker=_reranker(config_hash),
        candidate_k=settings.retrieval_top_k,
        context_k=settings.retrieval_context_k,
        min_relevance=settings.retrieval_min_relevance,
        domain_min_relevance=settings.retrieval_domain_min_relevance,
    )


def knowledge_search() -> ResilientKnowledgeSearch:
    effective = _effective()
    return _knowledge_search(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _knowledge_catalog(config_hash: str) -> KnowledgeCatalog:
    return KnowledgeCatalog(_object_store(config_hash))


def knowledge_catalog() -> KnowledgeCatalog:
    effective = _effective()
    return _knowledge_catalog(effective.config_hash)


@lru_cache(maxsize=_MAX_CONFIG_GENERATIONS)
def _conversation_agent(config_hash: str) -> ConversationAgent:
    return ConversationAgent(
        knowledge=_knowledge_search(config_hash),
        catalog=_knowledge_catalog(config_hash),
        model=_conversation_model(config_hash),
    )


def conversation_agent() -> ConversationAgent:
    """Canonical application agent, cached per immutable effective config hash."""
    effective = _effective()
    return _conversation_agent(effective.config_hash)
