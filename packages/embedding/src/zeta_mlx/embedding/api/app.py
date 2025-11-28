"""FastAPI 애플리케이션 팩토리"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zeta_mlx.core import Success, Failure, AppConfig, load_config
from zeta_mlx.embedding.engine import EmbeddingEngine, create_embedding_engine
from zeta_mlx.embedding.api.routes import (
    embeddings_router,
    health_router,
    set_embeddings_engine,
    set_health_engine,
)


def create_app(
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
    model_name: str | None = None,
    provider: str = "sentence-transformers",
) -> FastAPI:
    """
    FastAPI 앱 생성 (팩토리 패턴)

    Args:
        config: AppConfig 인스턴스 (직접 전달)
        config_path: YAML 설정 파일 경로
        model_name: 임베딩 모델 이름 (설정보다 우선)
        provider: 프로바이더 ("sentence-transformers")
    """
    # 설정 로드 (우선순위: config > config_path > 기본값)
    if config is None:
        if config_path:
            config_result = load_config(config_path)
            if isinstance(config_result, Failure):
                raise ValueError(f"Failed to load config: {config_result.error}")
            config = config_result.value
        else:
            # 기본 경로에서 자동 로드 시도
            config_result = load_config()
            config = config_result.value if isinstance(config_result, Success) else AppConfig()

    # 모델 이름 결정 (인자 > 설정 기본값)
    if model_name is None:
        default_alias = config.embedding_models.default
        default_model = config.embedding_models.get_default_model()
        model_name = default_model.path
        provider = default_model.provider

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """앱 생명주기 관리"""
        # Startup: 엔진 초기화
        engine_result = create_embedding_engine(model_name, provider)

        match engine_result:
            case Success(engine):
                app.state.engine = engine
                app.state.config = config
                set_embeddings_engine(engine)
                set_health_engine(engine)
            case Failure(error):
                raise RuntimeError(f"Failed to initialize embedding engine: {error}")

        yield  # 앱 실행

        # Shutdown: 정리 (필요시)

    app = FastAPI(
        title="Zeta MLX Embedding Server",
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


def create_app_from_config(config: AppConfig) -> FastAPI:
    """설정에서 앱 생성"""
    return create_app(config=config)


def create_app_with_engine(engine: EmbeddingEngine) -> FastAPI:
    """이미 생성된 엔진으로 앱 생성 (테스트용)"""
    app = FastAPI(
        title="Zeta MLX Embedding Server",
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
