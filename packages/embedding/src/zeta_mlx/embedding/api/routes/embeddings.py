"""임베딩 API 라우트"""
from fastapi import APIRouter, HTTPException

from zeta_mlx.core import Success, Failure
from zeta_mlx.embedding.engine import EmbeddingEngine
from zeta_mlx.embedding.types import EmbeddingRequest
from zeta_mlx.embedding.api.dto import (
    EmbeddingRequestDTO,
    EmbeddingResponseDTO,
    EmbeddingDataDTO,
    UsageDTO,
)

router = APIRouter(prefix="/v1", tags=["embeddings"])

# 모듈 레벨 엔진 (DI)
_engine: EmbeddingEngine | None = None


def set_engine(engine: EmbeddingEngine) -> None:
    """엔진 주입"""
    global _engine
    _engine = engine


def get_engine() -> EmbeddingEngine:
    """엔진 반환"""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Embedding engine not initialized")
    return _engine


# ============================================================
# DTO <-> Domain 변환 (Anti-Corruption Layer)
# ============================================================

def to_domain(dto: EmbeddingRequestDTO) -> EmbeddingRequest | None:
    """DTO -> Domain 변환"""
    result = EmbeddingRequest.create(
        model=dto.model,
        input=dto.to_input_list(),
        normalize=True,
    )
    match result:
        case Success(request):
            return request
        case Failure(_):
            return None


def to_dto(response, model: str) -> EmbeddingResponseDTO:
    """Domain -> DTO 변환"""
    return EmbeddingResponseDTO(
        data=[
            EmbeddingDataDTO(
                index=emb.index,
                embedding=list(emb.values),
            )
            for emb in response.embeddings
        ],
        model=model,
        usage=UsageDTO(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        ),
    )


# ============================================================
# 엔드포인트
# ============================================================

@router.post("/embeddings", response_model=EmbeddingResponseDTO)
async def create_embeddings(request: EmbeddingRequestDTO) -> EmbeddingResponseDTO:
    """
    OpenAI 호환 임베딩 생성

    POST /v1/embeddings
    """
    engine = get_engine()

    # DTO -> Domain
    domain_request = to_domain(request)
    if domain_request is None:
        raise HTTPException(status_code=400, detail="Invalid input: input cannot be empty")

    # 워크플로우 실행
    result = engine.embed(domain_request)

    match result:
        case Success(response):
            return to_dto(response, engine.model_name)
        case Failure(error):
            raise HTTPException(status_code=500, detail=str(error))
