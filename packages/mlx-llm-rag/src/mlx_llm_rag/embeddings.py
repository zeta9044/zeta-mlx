"""임베딩 생성 (I/O 경계)"""
from typing import Sequence, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from mlx_llm_core import Result, Success, Failure, ModelLoadError
from mlx_llm_rag.types import EmbeddingFn

if TYPE_CHECKING:
    from mlx_llm_core import EmbeddingConfig


def create_simple_embedding_fn(dimension: int = 384) -> EmbeddingFn:
    """간단한 해시 기반 임베딩 (테스트/개발용)

    실제 프로덕션에서는 sentence-transformers 사용 권장
    """
    def embed(texts: Sequence[str]) -> NDArray[np.float32]:
        embeddings = []
        for text in texts:
            # 해시 기반 의사 임베딩
            np.random.seed(hash(text) % (2**32))
            vec = np.random.randn(dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # 정규화
            embeddings.append(vec)
        return np.array(embeddings)

    return embed


def create_sentence_transformer_fn(
    model_name: str = "all-MiniLM-L6-v2"
) -> Result[EmbeddingFn, ModelLoadError]:
    """SentenceTransformer 기반 임베딩 함수 생성"""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        def embed(texts: Sequence[str]) -> NDArray[np.float32]:
            return model.encode(list(texts), convert_to_numpy=True)

        return Success(embed)

    except ImportError:
        return Failure(ModelLoadError(
            model_name=model_name,
            reason="sentence-transformers not installed. Install with: pip install sentence-transformers",
        ))
    except Exception as e:
        return Failure(ModelLoadError(
            model_name=model_name,
            reason=str(e),
        ))


# ============================================================
# 함수형 유틸리티
# ============================================================

def batch_embed(
    embed_fn: EmbeddingFn,
    texts: Sequence[str],
    batch_size: int = 32,
) -> NDArray[np.float32]:
    """배치 단위 임베딩 생성"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_fn(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """코사인 유사도 계산"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_batch(
    query: NDArray[np.float32],
    vectors: NDArray[np.float32],
) -> NDArray[np.float32]:
    """배치 코사인 유사도 계산"""
    # query: (d,), vectors: (n, d)
    query_norm = query / np.linalg.norm(query)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.dot(vectors_norm, query_norm)


# ============================================================
# Config 기반 팩토리
# ============================================================

def create_embedding_fn_from_config(
    config: "EmbeddingConfig",
) -> Result[EmbeddingFn, ModelLoadError]:
    """
    EmbeddingConfig에서 임베딩 함수 생성

    config.yaml 예시:
        rag:
          embedding:
            provider: sentence-transformers
            model_name: all-MiniLM-L6-v2
            dimension: 384
    """
    if config.provider == "simple":
        return Success(create_simple_embedding_fn(config.dimension))

    elif config.provider == "sentence-transformers":
        return create_sentence_transformer_fn(config.model_name)

    else:
        return Failure(ModelLoadError(
            model_name=config.provider,
            reason=f"Unknown embedding provider: {config.provider}",
        ))
