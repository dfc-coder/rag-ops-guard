from __future__ import annotations

from functools import lru_cache

from rag_ops_guard.config import Settings
from rag_ops_guard.control_plane import ControlPlaneConfig, fetch_control_plane


@lru_cache(maxsize=1)
def runtime_control_plane() -> ControlPlaneConfig:
    """Resolve platform discovery through AppConfig using the AWS SDK provider chain."""

    return fetch_control_plane()


def settings_from_control_plane(control_plane: ControlPlaneConfig) -> Settings:
    """Build the application settings model from trusted, non-secret control-plane values."""

    return Settings.model_validate(
        {
            "app_env": "runtime",
            "s3_document_bucket": control_plane.resources.document_bucket,
            "s3_vector_bucket": control_plane.resources.vector_bucket,
            "s3_vector_index": control_plane.resources.vector_index_base,
            "vector_dimension": control_plane.services.embedding.dimension,
            "llm_base_url": control_plane.services.llm.base_url,
            "llm_model": control_plane.services.llm.model,
            "embedding_base_url": control_plane.services.embedding.base_url,
            "embedding_model": control_plane.services.embedding.model,
            "embedding_dimension": control_plane.services.embedding.dimension,
            "reranker_base_url": control_plane.services.reranker.base_url,
            "reranker_model": control_plane.services.reranker.model,
        }
    )


def runtime_settings() -> Settings:
    return settings_from_control_plane(runtime_control_plane())


def reset_runtime_control_plane_cache() -> None:
    runtime_control_plane.cache_clear()
