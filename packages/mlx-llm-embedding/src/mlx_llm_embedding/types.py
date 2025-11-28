"""Embedding 도메인 타입 정의 (불변, 검증됨)"""
from dataclasses import dataclass
from typing import NewType, Self, Sequence
import numpy as np
from numpy.typing import NDArray

from mlx_llm_core import Result, Success, Failure, NonEmptyList


# ============================================================
# Constrained Types
# ============================================================

ModelName = NewType('ModelName', str)
Dimension = NewType('Dimension', int)


@dataclass(frozen=True)
class BatchSize:
    """1 ~ 2048 범위의 배치 크기"""
    value: int

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 2048:
            raise ValueError(f"BatchSize must be 1-2048, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(32)

    @classmethod
    def create(cls, value: int) -> Result['BatchSize', str]:
        if not 1 <= value <= 2048:
            return Failure(f"BatchSize must be 1-2048, got {value}")
        return Success(cls(value))


# ============================================================
# AND Types (Product Types)
# ============================================================

@dataclass(frozen=True)
class EmbeddingVector:
    """임베딩 벡터 (정규화됨)"""
    index: int
    values: tuple[float, ...]

    @classmethod
    def from_array(cls, index: int, arr: NDArray[np.float32]) -> Self:
        """numpy 배열에서 생성"""
        return cls(index=index, values=tuple(arr.tolist()))

    def to_array(self) -> NDArray[np.float32]:
        """numpy 배열로 변환"""
        return np.array(self.values, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class EmbeddingParams:
    """임베딩 생성 파라미터"""
    normalize: bool = True
    batch_size: BatchSize = BatchSize.default()

    @classmethod
    def default(cls) -> Self:
        return cls()


@dataclass(frozen=True)
class TokenUsage:
    """토큰 사용량"""
    prompt_tokens: int
    total_tokens: int

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> Self:
        """텍스트에서 대략적인 토큰 수 추정"""
        total = sum(len(t.split()) for t in texts)
        return cls(prompt_tokens=total, total_tokens=total)


# ============================================================
# Request/Response Types
# ============================================================

@dataclass(frozen=True)
class EmbeddingRequest:
    """임베딩 요청 (검증됨)"""
    model: str
    input: NonEmptyList[str]
    params: EmbeddingParams = EmbeddingParams.default()

    @classmethod
    def create(
        cls,
        model: str,
        input: list[str],
        normalize: bool = True,
    ) -> Result['EmbeddingRequest', str]:
        """검증된 요청 생성"""
        input_result = NonEmptyList.of(input)
        match input_result:
            case Failure(e):
                return Failure(f"Invalid input: {e}")
            case Success(non_empty):
                return Success(cls(
                    model=model,
                    input=non_empty,
                    params=EmbeddingParams(normalize=normalize),
                ))


@dataclass(frozen=True)
class EmbeddingResponse:
    """임베딩 응답"""
    embeddings: tuple[EmbeddingVector, ...]
    model: str
    usage: TokenUsage

    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return self.embeddings[0].dimension if self.embeddings else 0


# ============================================================
# Model Bundle
# ============================================================

@dataclass(frozen=True)
class EmbeddingModelInfo:
    """임베딩 모델 정보"""
    name: str
    dimension: int
    max_seq_length: int
    description: str = ""
