"""Models 라우트"""
from fastapi import APIRouter
from mlx_llm_api.dto.responses import ModelsResponseDTO
from mlx_llm_api.converters import create_models_response

router = APIRouter(prefix="/v1", tags=["models"])

# 사용 가능한 모델 목록 (의존성 주입으로 설정)
_available_models: list[str] = []


def set_available_models(models: list[str]) -> None:
    """사용 가능한 모델 설정"""
    global _available_models
    _available_models = models


def get_available_models() -> list[str]:
    """사용 가능한 모델 조회"""
    return _available_models


@router.get("/models", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """사용 가능한 모델 목록 조회"""
    return create_models_response(_available_models)
