"""MLX LLM Inference - MLX Integration Layer"""
from zeta_mlx_inference.engine import (
    InferenceEngine,
    GenerateFn, StreamFn, TokenCountFn, TemplateFn,
)
from zeta_mlx_inference.manager import (
    ModelManager, LoadedModel,
    create_model_manager, create_model_manager_from_yaml,
)
from zeta_mlx_inference.loader import (
    ModelBundle,
    load_model, load_model_safe, unload_model,
)
from zeta_mlx_inference.streaming import (
    mlx_stream_generator, chunk_stream,
)

__version__ = "0.1.0"

__all__ = [
    # Engine (단일 모델)
    "InferenceEngine",
    # Manager (다중 모델)
    "ModelManager",
    "LoadedModel",
    "create_model_manager",
    "create_model_manager_from_yaml",
    # Types
    "GenerateFn", "StreamFn", "TokenCountFn", "TemplateFn",
    # Loader
    "ModelBundle",
    "load_model", "load_model_safe", "unload_model",
    # Streaming
    "mlx_stream_generator", "chunk_stream",
]
