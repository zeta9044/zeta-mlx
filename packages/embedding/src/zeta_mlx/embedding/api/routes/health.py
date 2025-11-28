"""헬스체크 라우트"""
from fastapi import APIRouter
from pydantic import BaseModel

from zeta_mlx.embedding.engine import EmbeddingEngine
from zeta_mlx.embedding.api.dto import ModelInfoDTO, ModelsResponseDTO

router = APIRouter(tags=["health"])

# 모듈 레벨 엔진 (DI)
_engine: EmbeddingEngine | None = None


def set_engine(engine: EmbeddingEngine) -> None:
    """엔진 주입"""
    global _engine
    _engine = engine


class HealthResponse(BaseModel):
    """헬스 응답"""
    status: str
    model: str | None = None
    dimension: int | None = None


@router.get("/health")
async def health() -> HealthResponse:
    """헬스체크"""
    if _engine is None:
        return HealthResponse(status="unhealthy")

    return HealthResponse(
        status="healthy",
        model=_engine.model_name,
        dimension=_engine.dimension,
    )


@router.get("/v1/models", response_model=ModelsResponseDTO)
async def list_models() -> ModelsResponseDTO:
    """모델 목록"""
    if _engine is None:
        return ModelsResponseDTO(data=[])

    return ModelsResponseDTO(
        data=[
            ModelInfoDTO(
                id=_engine.model_name,
                dimension=_engine.dimension,
                max_seq_length=_engine.info.max_seq_length,
            )
        ]
    )
