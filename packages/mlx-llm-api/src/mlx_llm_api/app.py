"""FastAPI 애플리케이션 팩토리 (다중 모델 지원)"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlx_llm_core import AppConfig, load_config, Success, Failure
from mlx_llm_inference import ModelManager, create_model_manager
from mlx_llm_api.routes import chat_router, models_router, health_router
from mlx_llm_api.routes.chat import set_model_manager
from mlx_llm_api.routes.models import set_model_manager as set_models_manager
from mlx_llm_api.routes.health import set_model_manager as set_health_manager


def create_app(
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """
    FastAPI 앱 생성 (팩토리 패턴)

    Args:
        config: AppConfig 인스턴스 (직접 전달)
        config_path: YAML 설정 파일 경로
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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """앱 생명주기 관리"""
        # Startup: 모델 관리자 초기화
        model_manager = create_model_manager(config.models)
        app.state.model_manager = model_manager
        app.state.config = config

        # 라우트에 모델 관리자 주입
        set_model_manager(model_manager)
        set_models_manager(model_manager)
        set_health_manager(model_manager)

        # 기본 모델 미리 로드 (선택적)
        preload_models = [config.models.default]
        model_manager.preload(preload_models)

        yield  # 앱 실행

        # Shutdown: 모든 모델 언로드
        model_manager.unload_all()

    app = FastAPI(
        title="MLX LLM Server",
        description="OpenAI-compatible LLM inference server powered by MLX (Multi-Model)",
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


def create_app_from_yaml(config_path: str | Path) -> FastAPI:
    """YAML 설정 파일에서 앱 생성"""
    return create_app(config_path=config_path)


def create_app_with_manager(manager: ModelManager) -> FastAPI:
    """이미 생성된 모델 관리자로 앱 생성 (테스트용)"""
    app = FastAPI(
        title="MLX LLM Server",
        description="OpenAI-compatible LLM inference server powered by MLX (Multi-Model)",
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

    # 모델 관리자 설정
    app.state.model_manager = manager
    set_model_manager(manager)
    set_models_manager(manager)
    set_health_manager(manager)

    # 라우터 등록
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(chat_router)

    return app
