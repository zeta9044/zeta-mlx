"""Embedding 에러 타입 (OR Types / Sum Types)"""
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ModelLoadError:
    """모델 로드 실패"""
    model_name: str
    reason: str

    def __str__(self) -> str:
        return f"Failed to load model '{self.model_name}': {self.reason}"


@dataclass(frozen=True)
class EmbeddingError:
    """임베딩 생성 실패"""
    reason: str

    def __str__(self) -> str:
        return f"Embedding failed: {self.reason}"


@dataclass(frozen=True)
class ValidationError:
    """입력 검증 실패"""
    field: str
    message: str

    def __str__(self) -> str:
        return f"Validation error on '{self.field}': {self.message}"


@dataclass(frozen=True)
class BatchSizeExceededError:
    """배치 크기 초과"""
    actual: int
    limit: int

    def __str__(self) -> str:
        return f"Batch size {self.actual} exceeds limit {self.limit}"


# Union type for all embedding errors
EmbeddingServiceError = Union[
    ModelLoadError,
    EmbeddingError,
    ValidationError,
    BatchSizeExceededError,
]
