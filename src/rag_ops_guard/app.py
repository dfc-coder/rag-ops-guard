from functools import lru_cache

from rag_ops_guard.adapters.aws.s3_store import S3ObjectStore
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm.llamacpp_chat import LlamaCppChatAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.adapters.reranking.llamacpp_reranker import LlamaCppRerankerAdapter
from rag_ops_guard.agent.catalog import KnowledgeCatalog
from rag_ops_guard.agent.router import SemanticRouter
from rag_ops_guard.config import get_settings
from rag_ops_guard.graph.conversational_agent import ConversationalAgent
from rag_ops_guard.graph.prompts import CONVERSATIONAL_SYSTEM_PROMPT, GROUNDING_SYSTEM_PROMPT
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.observability.langsmith import configure_langsmith
from rag_ops_guard.retrieval.resilient import ResilientKnowledgeSearch
from rag_ops_guard.retrieval.resolver import EvidenceResolver


@lru_cache(maxsize=1)
def object_store() -> S3ObjectStore:
    settings = get_settings()
    return S3ObjectStore(
        bucket=settings.s3_document_bucket,
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
    )


@lru_cache(maxsize=1)
def embeddings() -> LlamaCppEmbeddingAdapter:
    settings = get_settings()
    return LlamaCppEmbeddingAdapter(
        settings.embedding_base_url,
        settings.embedding_model,
        settings.embedding_dimension,
    )


@lru_cache(maxsize=1)
def reranker() -> LlamaCppRerankerAdapter:
    settings = get_settings()
    return LlamaCppRerankerAdapter(
        settings.reranker_base_url,
        settings.reranker_model,
        settings.reranker_timeout_seconds,
    )


@lru_cache(maxsize=1)
def vector_store() -> S3VectorsStore:
    settings = get_settings()
    return S3VectorsStore(
        vector_bucket=settings.s3_vector_bucket,
        index_name=settings.s3_vector_index,
        object_store=object_store(),
        endpoint_url=settings.aws_endpoint_url,
        region_name=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key,
    )


@lru_cache(maxsize=1)
def chat_model() -> LlamaCppChatAdapter:
    settings = get_settings()
    configure_langsmith(settings)
    return LlamaCppChatAdapter(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        analysis_max_tokens=settings.llm_analysis_max_tokens,
        answer_max_tokens=settings.llm_answer_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
        top_p=settings.llm_top_p,
        top_k=settings.llm_top_k,
        min_p=settings.llm_min_p,
        presence_penalty=settings.llm_presence_penalty,
        repeat_penalty=settings.llm_repeat_penalty,
        answer_system_prompt=GROUNDING_SYSTEM_PROMPT,
        chat_system_prompt=CONVERSATIONAL_SYSTEM_PROMPT,
    )


@lru_cache(maxsize=1)
def ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        object_store=object_store(),
        vector_store=vector_store(),
        embeddings=embeddings(),
        chunker=MarkdownChunker(
            # Token counting is lightweight orchestration work and stays on the CPU-side
            # llama.cpp server. OpenVINO owns only embeddings and reranking; its v3 API is
            # not wire-compatible with llama.cpp's /tokenize endpoint used by this adapter.
            token_counter=LlamaCppTokenCounter(settings.llm_base_url),
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
        ),
    )


@lru_cache(maxsize=1)
def knowledge_search() -> ResilientKnowledgeSearch:
    settings = get_settings()
    return ResilientKnowledgeSearch(
        embeddings=embeddings(),
        vectors=vector_store(),
        objects=object_store(),
        resolver=EvidenceResolver(),
        reranker=reranker(),
        candidate_k=settings.retrieval_top_k,
        context_k=settings.retrieval_context_k,
    )


@lru_cache(maxsize=1)
def knowledge_catalog() -> KnowledgeCatalog:
    return KnowledgeCatalog(object_store())


@lru_cache(maxsize=1)
def query_workflow() -> ConversationalAgent:
    settings = get_settings()
    configure_langsmith(settings)
    return ConversationalAgent(
        chat=chat_model(),
        router=SemanticRouter(
            embeddings(),
            min_score=settings.router_min_score,
            min_margin=settings.router_min_margin,
        ),
        knowledge=knowledge_search(),
        catalog=knowledge_catalog(),
    )
