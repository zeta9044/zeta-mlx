"""FastAPI 애플리케이션 팩토리"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlx_llm_core import ServerConfig
from mlx_llm_inference import InferenceEngine, load_model_safe
from mlx_llm_api.routes import chat_router, models_router, health_router
from mlx_llm_api.routes.chat import set_inference_engine
from mlx_llm_api.routes.models import set_available_models
from mlx_llm_api.routes.health import set_current_model


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """FastAPI 앱 생성 (팩토리 패턴)"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """앱 생명주기 관리"""
        # Startup
        if config and config.model_name:
            # 모델 로드
            bundle_result = load_model_safe(config.model_name)
            if hasattr(bundle_result, "value"):
                bundle = bundle_result.value
                engine = InferenceEngine(bundle)
                set_inference_engine(engine)
                set_current_model(config.model_name)
                set_available_models([config.model_name])

        yield  # 앱 실행

        # Shutdown
        pass

    app = FastAPI(
        title="MLX LLM Server",
        description="OpenAI-compatible LLM inference server powered by MLX",
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
    app.include_router(models_router)
    app.include_router(chat_router)

    return app


def create_app_with_engine(engine: InferenceEngine, model_name: str) -> FastAPI:
    """이미 로드된 엔진으로 앱 생성 (테스트용)"""
    app = FastAPI(
        title="MLX LLM Server",
        description="OpenAI-compatible LLM inference server powered by MLX",
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

    # 엔진 및 모델 설정
    set_inference_engine(engine)
    set_current_model(model_name)
    set_available_models([model_name])

    # 라우터 등록
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)

    return app
