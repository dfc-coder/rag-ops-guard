from functools import lru_cache

from rag_ops_guard.adapters.aws.s3_store import S3ObjectStore
from rag_ops_guard.adapters.aws.s3_vectors import S3VectorsStore
from rag_ops_guard.adapters.embeddings.llamacpp_embeddings import LlamaCppEmbeddingAdapter
from rag_ops_guard.adapters.llm.llamacpp_chat import LlamaCppChatAdapter
from rag_ops_guard.adapters.llm.tokenizer import LlamaCppTokenCounter
from rag_ops_guard.config import get_settings
from rag_ops_guard.graph.timed_workflow import TimedRagWorkflow
from rag_ops_guard.ingestion.chunker import MarkdownChunker
from rag_ops_guard.ingestion.service import IngestionService
from rag_ops_guard.observability.langsmith import configure_langsmith
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
    )


@lru_cache(maxsize=1)
def ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        object_store=object_store(),
        vector_store=vector_store(),
        embeddings=embeddings(),
        chunker=MarkdownChunker(
            token_counter=LlamaCppTokenCounter(settings.embedding_base_url),
            target_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap,
        ),
    )


@lru_cache(maxsize=1)
def query_workflow() -> TimedRagWorkflow:
    settings = get_settings()
    configure_langsmith(settings)
    return TimedRagWorkflow(
        chat=chat_model(),
        embeddings=embeddings(),
        vectors=vector_store(),
        resolver=EvidenceResolver(),
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_context_k=settings.retrieval_context_k,
    )
