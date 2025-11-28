"""Models 라우트 (다중 모델 지원)"""
import time
from fastapi import APIRouter, HTTPException
from zeta_mlx.core import Failure
from zeta_mlx.inference import ModelManager
from zeta_mlx.inference.api.dto.responses import ModelsResponseDTO
from zeta_mlx.inference.api.converters import create_model_dto, create_models_response

router = APIRouter(prefix="/v1", tags=["models"])

# 모델 관리자 (의존성 주입으로 설정)
_model_manager: ModelManager | None = None


def set_model_manager(manager: ModelManager) -> None:
    """모델 관리자 설정"""
    global _model_manager
    _model_manager = manager


def get_model_manager() -> ModelManager | None:
    """모델 관리자 조회"""
    return _model_manager


@router.get("/models", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """
    사용 가능한 모델 목록 조회

    config.yaml에 정의된 모든 모델을 반환합니다.
    """
    if _model_manager is None:
        return create_models_response([])

    return create_models_response(_model_manager.list_available())


@router.get("/models/loaded")
async def list_loaded_models():
    """
    현재 메모리에 로드된 모델 목록

    LRU 캐시에 있는 모델들을 반환합니다.
    """
    if _model_manager is None:
        return {"loaded": [], "default": ""}

    return {
        "loaded": _model_manager.list_loaded(),
        "default": _model_manager.default_alias,
    }


@router.get("/models/{model_alias}")
async def get_model_info(model_alias: str):
    """
    특정 모델 정보 조회
    """
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    info = _model_manager.get_model_info(model_alias)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_alias}' not found")

    return {
        "id": model_alias,
        "path": info.path,
        "context_length": info.context_length,
        "quantization": info.quantization,
        "description": info.description,
        "loaded": model_alias in _model_manager.list_loaded(),
    }


@router.post("/models/{model_alias}/load")
async def load_model(model_alias: str):
    """
    모델 명시적 로드

    사용 전 미리 로드하여 첫 요청 지연을 줄입니다.
    """
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    result = _model_manager.get_engine(model_alias)

    if isinstance(result, Failure):
        return {"success": False, "error": str(result.error), "model": model_alias}

    return {"success": True, "model": model_alias}


@router.post("/models/{model_alias}/unload")
async def unload_model(model_alias: str):
    """
    모델 명시적 언로드

    메모리에서 모델을 제거합니다.
    """
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    success = _model_manager.unload(model_alias)

    return {"success": success, "model": model_alias}
