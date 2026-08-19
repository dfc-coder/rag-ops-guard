from __future__ import annotations

from dataclasses import dataclass

from rag_ops_guard.adapters.aws.s3_store import S3ObjectStore
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm.openai_tool_calling import OpenAIToolCallingAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter
from rag_ops_guard.agent.catalog import KnowledgeCatalog
from rag_ops_guard.agent.conversation import ConversationAgent
from rag_ops_guard.configstore.runtime import EffectiveConfig, resolve_effective_config
from rag_ops_guard.container import Container
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.observability.langsmith import configure_langsmith
from rag_ops_guard.retrieval.resilient import ResilientKnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from rag_ops_guard.tenancy import KeyLayout, RequestContext

_MAX_TENANT_CONFIG_GENERATIONS = 32
_DEFAULT_CONTEXT = RequestContext(principal="system:default", tenant_id="default")


@dataclass(frozen=True, slots=True)
class DependencyBundle:
    context: RequestContext
    config_hash: str
    key_layout: KeyLayout
    objects: S3ObjectStore
    embeddings: LlamaCppEmbeddingAdapter
    reranker: LlamaCppRerankerAdapter
    vectors: S3VectorsStore
    model: OpenAIToolCallingAdapter
    ingestion: IngestionService
    knowledge: ResilientKnowledgeSearch
    catalog: KnowledgeCatalog
    agent: ConversationAgent


_CONTAINER: Container[DependencyBundle] = Container(
    max_generations=_MAX_TENANT_CONFIG_GENERATIONS
)


def _build_dependencies(context: RequestContext, effective: EffectiveConfig) -> DependencyBundle:
    settings = effective.settings
    keys = KeyLayout(context.tenant_id)
    objects = S3ObjectStore(
        bucket=settings.s3_document_bucket,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
    )
    embeddings_adapter = LlamaCppEmbeddingAdapter(
        settings.embedding_base_url,
        settings.embedding_model,
        settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    reranker_adapter = LlamaCppRerankerAdapter(
        settings.reranker_base_url,
        settings.reranker_model,
        settings.reranker_timeout_seconds,
    )
    vectors = S3VectorsStore(
        vector_bucket=settings.s3_vector_bucket,
        index_name=keys.vector_index(settings.s3_vector_index),
        object_store=objects,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        key_layout=keys,
    )
    configure_langsmith(settings)
    model = OpenAIToolCallingAdapter(
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
    ingestion = IngestionService(
        object_store=objects,
        vector_store=vectors,
        embeddings=embeddings_adapter,
        chunker=MarkdownChunker(
            token_counter=LlamaCppTokenCounter(settings.llm_base_url),
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
        ),
        key_layout=keys,
    )
    knowledge = ResilientKnowledgeSearch(
        embeddings=embeddings_adapter,
        vectors=vectors,
        objects=objects,
        resolver=EvidenceResolver(),
        reranker=reranker_adapter,
        candidate_k=settings.retrieval_top_k,
        context_k=settings.retrieval_context_k,
        min_relevance=settings.retrieval_min_relevance,
        domain_min_relevance=settings.retrieval_domain_min_relevance,
    )
    catalog = KnowledgeCatalog(objects, keys)
    agent = ConversationAgent(knowledge=knowledge, catalog=catalog, model=model)
    return DependencyBundle(
        context=context,
        config_hash=effective.config_hash,
        key_layout=keys,
        objects=objects,
        embeddings=embeddings_adapter,
        reranker=reranker_adapter,
        vectors=vectors,
        model=model,
        ingestion=ingestion,
        knowledge=knowledge,
        catalog=catalog,
        agent=agent,
    )


def dependencies(context: RequestContext | None = None) -> DependencyBundle:
    """Resolve one warm dependency graph per authenticated tenant/config generation."""
    request_context = context or _DEFAULT_CONTEXT
    effective = resolve_effective_config()
    return _CONTAINER.get(
        request_context,
        effective.config_hash,
        lambda resolved_context, _: _build_dependencies(resolved_context, effective),
    )


def object_store(context: RequestContext | None = None) -> S3ObjectStore:
    return dependencies(context).objects


def embeddings(context: RequestContext | None = None) -> LlamaCppEmbeddingAdapter:
    return dependencies(context).embeddings


def reranker(context: RequestContext | None = None) -> LlamaCppRerankerAdapter:
    return dependencies(context).reranker


def vector_store(context: RequestContext | None = None) -> S3VectorsStore:
    return dependencies(context).vectors


def conversation_model(context: RequestContext | None = None) -> OpenAIToolCallingAdapter:
    return dependencies(context).model


def ingestion_service(context: RequestContext | None = None) -> IngestionService:
    return dependencies(context).ingestion


def knowledge_search(context: RequestContext | None = None) -> ResilientKnowledgeSearch:
    return dependencies(context).knowledge


def knowledge_catalog(context: RequestContext | None = None) -> KnowledgeCatalog:
    return dependencies(context).catalog


def conversation_agent(context: RequestContext | None = None) -> ConversationAgent:
    return dependencies(context).agent


def clear_dependency_container() -> None:
    """Test/reload hook; production lifecycle uses bounded LRU eviction."""
    _CONTAINER.clear()
