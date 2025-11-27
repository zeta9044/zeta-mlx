"""모델 로더"""
from dataclasses import dataclass
from typing import Any
from functools import lru_cache
import sys

from mlx_llm_core import Result, Success, Failure, ModelNotFoundError


@dataclass(frozen=True)
class ModelBundle:
    """로드된 모델 번들"""
    name: str
    model: Any  # MLX model
    tokenizer: Any  # Tokenizer


def setup_custom_models() -> None:
    """커스텀 모델 등록"""
    from mlx_llm_inference.custom_models import qwen3

    # MLX-LM이 찾을 수 있도록 등록
    sys.modules['mlx_lm.models.qwen3'] = qwen3


@lru_cache(maxsize=4)
def load_model(model_name: str) -> ModelBundle:
    """
    모델 로드 (캐시됨)

    최대 4개 모델까지 메모리에 유지합니다.
    """
    setup_custom_models()

    from mlx_lm import load

    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {model_name}")

    return ModelBundle(
        name=model_name,
        model=model,
        tokenizer=tokenizer,
    )


def load_model_safe(model_name: str) -> Result[ModelBundle, ModelNotFoundError]:
    """모델 로드 (Result 반환)"""
    try:
        bundle = load_model(model_name)
        return Success(bundle)
    except Exception:
        return Failure(ModelNotFoundError(model=model_name))


def unload_model(model_name: str) -> None:
    """모델 언로드 (캐시에서 제거)"""
    load_model.cache_clear()


def list_loaded_models() -> list[str]:
    """로드된 모델 목록"""
    # lru_cache doesn't expose keys directly
    return []
