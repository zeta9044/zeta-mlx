"""FastAPI 애플리케이션 팩토리"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zeta_mlx_core import Success, Failure
from zeta_mlx_embedding.engine import EmbeddingEngine, create_embedding_engine
from zeta_mlx_embedding.api.routes import (
    embeddings_router,
    health_router,
    set_embeddings_engine,
    set_health_engine,
)


def create_app(
    model_name: str = "all-MiniLM-L6-v2",
    provider: str = "sentence-transformers",
) -> FastAPI:
    """
    FastAPI 앱 생성 (팩토리 패턴)

    Args:
        model_name: 임베딩 모델 이름
        provider: 프로바이더 ("sentence-transformers")
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """앱 생명주기 관리"""
        # Startup: 엔진 초기화
        engine_result = create_embedding_engine(model_name, provider)

        match engine_result:
            case Success(engine):
                app.state.engine = engine
                set_embeddings_engine(engine)
                set_health_engine(engine)
            case Failure(error):
                raise RuntimeError(f"Failed to initialize embedding engine: {error}")

        yield  # 앱 실행

        # Shutdown: 정리 (필요시)

    app = FastAPI(
        title="MLX Embedding Server",
        description="OpenAI-compatible embedding server",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS 미들웨어
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(health_router)
    app.include_router(embeddings_router)

    return app


def create_app_with_engine(engine: EmbeddingEngine) -> FastAPI:
    """이미 생성된 엔진으로 앱 생성 (테스트용)"""
    app = FastAPI(
        title="MLX Embedding Server",
        description="OpenAI-compatible embedding server",
        version="0.1.0",
    )

    # CORS 미들웨어
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 엔진 설정
    app.state.engine = engine
    set_embeddings_engine(engine)
    set_health_engine(engine)

    # 라우터 등록
    app.include_router(health_router)
    app.include_router(embeddings_router)

    return app
