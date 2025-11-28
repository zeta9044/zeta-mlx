"""임베딩 엔진 (함수형 워크플로우)"""
from typing import Callable, Sequence
import numpy as np
from numpy.typing import NDArray

from zeta_mlx.core import Result, Success, Failure, Railway
from zeta_mlx.embedding.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    EmbeddingParams,
    TokenUsage,
)
from zeta_mlx.embedding.errors import (
    EmbeddingError,
    EmbeddingServiceError,
    BatchSizeExceededError,
)
from zeta_mlx.embedding.loader import EmbeddingBundle


# ============================================================
# 함수 타입 정의
# ============================================================

EmbedFn = Callable[[Sequence[str]], Result[NDArray[np.float32], EmbeddingError]]
NormalizeFn = Callable[[NDArray[np.float32]], NDArray[np.float32]]


# ============================================================
# 순수 함수 (Pure)
# ============================================================

def normalize_embeddings(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2 정규화"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # 0 나눗셈 방지
    return embeddings / norms


def validate_batch_size(
    texts: list[str],
    max_batch: int = 2048,
) -> Result[list[str], BatchSizeExceededError]:
    """배치 크기 검증"""
    if len(texts) > max_batch:
        return Failure(BatchSizeExceededError(actual=len(texts), limit=max_batch))
    return Success(texts)


def to_embedding_vectors(
    embeddings: NDArray[np.float32],
) -> tuple[EmbeddingVector, ...]:
    """numpy 배열을 EmbeddingVector 튜플로 변환"""
    return tuple(
        EmbeddingVector.from_array(i, emb)
        for i, emb in enumerate(embeddings)
    )


def create_response(
    embeddings: NDArray[np.float32],
    model: str,
    texts: list[str],
) -> EmbeddingResponse:
    """EmbeddingResponse 생성"""
    return EmbeddingResponse(
        embeddings=to_embedding_vectors(embeddings),
        model=model,
        usage=TokenUsage.from_texts(texts),
    )


# ============================================================
# 팩토리 함수 (Impure - I/O 경계)
# ============================================================

def create_sentence_transformer_embed(bundle: EmbeddingBundle) -> EmbedFn:
    """SentenceTransformer 기반 embed 함수 생성"""

    def embed(texts: Sequence[str]) -> Result[NDArray[np.float32], EmbeddingError]:
        try:
            embeddings = bundle.model.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return Success(embeddings.astype(np.float32))
        except Exception as e:
            return Failure(EmbeddingError(reason=str(e)))

    return embed


# ============================================================
# 워크플로우 (Railway Oriented)
# ============================================================

def embed_workflow(
    request: EmbeddingRequest,
    embed_fn: EmbedFn,
) -> Result[EmbeddingResponse, EmbeddingServiceError]:
    """
    임베딩 워크플로우

    1. 배치 크기 검증
    2. 임베딩 생성
    3. 정규화 (선택)
    4. 응답 생성
    """
    texts = request.input.to_list()

    return (
        Railway.of(texts)
        .bind(lambda t: validate_batch_size(t, request.params.batch_size.value * 64))
        .bind(embed_fn)
        .map(lambda e: normalize_embeddings(e) if request.params.normalize else e)
        .map(lambda e: create_response(e, request.model, texts))
        .unwrap()
    )


# ============================================================
# Embedding Engine (Facade)
# ============================================================

class EmbeddingEngine:
    """
    임베딩 엔진 (Facade)

    모델을 로드하고 임베딩 워크플로우를 제공합니다.
    """

    def __init__(self, bundle: EmbeddingBundle):
        self._bundle = bundle
        self._embed_fn = create_sentence_transformer_embed(bundle)

    def embed(
        self,
        request: EmbeddingRequest,
    ) -> Result[EmbeddingResponse, EmbeddingServiceError]:
        """임베딩 생성"""
        return embed_workflow(request, self._embed_fn)

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> Result[EmbeddingResponse, EmbeddingServiceError]:
        """간단한 텍스트 임베딩"""
        request_result = EmbeddingRequest.create(
            model=self._bundle.name,
            input=texts,
            normalize=normalize,
        )

        match request_result:
            case Failure(e):
                return Failure(EmbeddingError(reason=str(e)))
            case Success(request):
                return self.embed(request)

    @property
    def model_name(self) -> str:
        return self._bundle.name

    @property
    def dimension(self) -> int:
        return self._bundle.dimension

    @property
    def info(self):
        return self._bundle.info


# ============================================================
# 팩토리
# ============================================================

def create_embedding_engine(
    model_name: str = "all-MiniLM-L6-v2",
    provider: str = "sentence-transformers",
) -> Result[EmbeddingEngine, EmbeddingServiceError]:
    """임베딩 엔진 생성"""
    from zeta_mlx.embedding.loader import load_embedding_model

    return (
        Railway.from_result(load_embedding_model(model_name, provider))
        .map(EmbeddingEngine)
        .unwrap()
    )
