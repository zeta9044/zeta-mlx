"""임베딩 모델 로더 (I/O 경계)"""
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from zeta_mlx_core import Result, Success, Failure
from zeta_mlx_embedding.types import EmbeddingModelInfo
from zeta_mlx_embedding.errors import ModelLoadError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ============================================================
# Model Bundle (Impure - 실제 모델 인스턴스 보유)
# ============================================================

@dataclass
class EmbeddingBundle:
    """로드된 임베딩 모델 번들"""
    model: Any  # SentenceTransformer 또는 MLX 모델
    info: EmbeddingModelInfo
    _encode_fn: Any = None  # 캐시된 encode 함수

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def dimension(self) -> int:
        return self.info.dimension


# ============================================================
# SentenceTransformer 로더
# ============================================================

def load_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
) -> Result[EmbeddingBundle, ModelLoadError]:
    """
    SentenceTransformer 모델 로드

    Args:
        model_name: HuggingFace 모델 이름 또는 로컬 경로

    Returns:
        Result[EmbeddingBundle, ModelLoadError]
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        info = EmbeddingModelInfo(
            name=model_name,
            dimension=model.get_sentence_embedding_dimension(),
            max_seq_length=model.max_seq_length,
            description=f"SentenceTransformer: {model_name}",
        )

        return Success(EmbeddingBundle(model=model, info=info))

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
# 팩토리 함수
# ============================================================

SUPPORTED_PROVIDERS = ["sentence-transformers"]


def load_embedding_model(
    model_name: str,
    provider: str = "sentence-transformers",
) -> Result[EmbeddingBundle, ModelLoadError]:
    """
    임베딩 모델 로드 (팩토리)

    Args:
        model_name: 모델 이름
        provider: 프로바이더 ("sentence-transformers")

    Returns:
        Result[EmbeddingBundle, ModelLoadError]
    """
    match provider:
        case "sentence-transformers":
            return load_sentence_transformer(model_name)
        case _:
            return Failure(ModelLoadError(
                model_name=model_name,
                reason=f"Unknown provider: {provider}. Supported: {SUPPORTED_PROVIDERS}",
            ))
