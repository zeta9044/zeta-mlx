"""API 응답 DTO (외부 세계 - Pydantic)"""
from pydantic import BaseModel, Field


class EmbeddingDataDTO(BaseModel):
    """임베딩 데이터"""
    object: str = "embedding"
    index: int
    embedding: list[float]


class UsageDTO(BaseModel):
    """사용량"""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponseDTO(BaseModel):
    """OpenAI 호환 임베딩 응답"""
    object: str = "list"
    data: list[EmbeddingDataDTO]
    model: str
    usage: UsageDTO


class ErrorDTO(BaseModel):
    """에러 응답"""
    error: dict = Field(..., description="에러 정보")


class ModelInfoDTO(BaseModel):
    """모델 정보"""
    id: str
    object: str = "model"
    owned_by: str = "mlx-llm"
    dimension: int
    max_seq_length: int


class ModelsResponseDTO(BaseModel):
    """모델 목록 응답"""
    object: str = "list"
    data: list[ModelInfoDTO]
