"""API Routes"""
from mlx_llm_embedding.api.routes.embeddings import router as embeddings_router
from mlx_llm_embedding.api.routes.embeddings import set_engine as set_embeddings_engine
from mlx_llm_embedding.api.routes.health import router as health_router
from mlx_llm_embedding.api.routes.health import set_engine as set_health_engine

__all__ = [
    "embeddings_router",
    "health_router",
    "set_embeddings_engine",
    "set_health_engine",
]
