"""
Zeta MLX Embedding - 함수형 임베딩 서빙 패키지

OpenAI 호환 /v1/embeddings API를 제공합니다.

Example:
    # 엔진 직접 사용
    from zeta_mlx_embedding import create_embedding_engine, Success

    result = create_embedding_engine("all-MiniLM-L6-v2")
    match result:
        case Success(engine):
            response = engine.embed_texts(["Hello", "World"])
        case Failure(error):
            print(f"Error: {error}")

    # API 서버 실행
    from zeta_mlx_embedding.api import create_app
    import uvicorn

    app = create_app(model_name="all-MiniLM-L6-v2")
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

# Types
from zeta_mlx_embedding.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    EmbeddingParams,
    EmbeddingModelInfo,
    TokenUsage,
    BatchSize,
)

# Errors
from zeta_mlx_embedding.errors import (
    ModelLoadError,
    EmbeddingError,
    ValidationError,
    BatchSizeExceededError,
    EmbeddingServiceError,
)

# Engine
from zeta_mlx_embedding.engine import (
    EmbeddingEngine,
    create_embedding_engine,
    embed_workflow,
    EmbedFn,
)

# Loader
from zeta_mlx_embedding.loader import (
    EmbeddingBundle,
    load_embedding_model,
    load_sentence_transformer,
)

__all__ = [
    # Types
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingVector",
    "EmbeddingParams",
    "EmbeddingModelInfo",
    "TokenUsage",
    "BatchSize",
    # Errors
    "ModelLoadError",
    "EmbeddingError",
    "ValidationError",
    "BatchSizeExceededError",
    "EmbeddingServiceError",
    # Engine
    "EmbeddingEngine",
    "create_embedding_engine",
    "embed_workflow",
    "EmbedFn",
    # Loader
    "EmbeddingBundle",
    "load_embedding_model",
    "load_sentence_transformer",
]
