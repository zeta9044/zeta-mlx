"""Health Check 라우트"""
from fastapi import APIRouter
from mlx_llm_api.dto.responses import HealthResponseDTO
from mlx_llm_api.converters import create_health_response

router = APIRouter(tags=["health"])

# 버전 정보 (패키지에서 가져올 수 있음)
VERSION = "0.1.0"

# 현재 로드된 모델 (의존성 주입으로 설정)
_current_model: str | None = None


def set_current_model(model: str | None) -> None:
    """현재 모델 설정 (앱 시작 시 호출)"""
    global _current_model
    _current_model = model


def get_current_model() -> str | None:
    """현재 모델 조회"""
    return _current_model


@router.get("/health", response_model=HealthResponseDTO)
async def health_check() -> HealthResponseDTO:
    """헬스 체크 엔드포인트"""
    return create_health_response(
        status="healthy",
        model=_current_model,
        version=VERSION,
    )
