"""Zeta MLX Core - Pure Domain Layer"""
from zeta_mlx.core.types import (
    # Constrained Types
    Role, ModelName, TokenCount,
    Temperature, TopP, MaxTokens,
    # Domain Types
    Message, GenerationParams,
    ToolFunction, ToolDefinition, ToolCall,
    NonEmptyList,
    ChatRequest, InferenceRequest, InferenceResponse, TokenUsage,
)
from zeta_mlx.core.result import (
    Result, Success, Failure,
    Railway,
    map_result, bind, map_error, tee,
    unwrap_or, unwrap_or_else,
    validate_all,
)
from zeta_mlx.core.errors import (
    ValidationError, TokenLimitError,
    ModelNotFoundError, GenerationError,
    InferenceError,
    error_to_dict,
)
from zeta_mlx.core.validation import (
    validate_messages, validate_params, check_token_limit,
)
from zeta_mlx.core.pipeline import (
    pipe, compose, identity, const, curry2, flip,
)
from zeta_mlx.core.config import (
    ServerConfig, EmbeddingServerConfig,
    ModelDefinition, ModelsConfig,
    EmbeddingModelDefinition, EmbeddingModelsConfig,
    InferenceConfig, AppConfig,
    EmbeddingConfig, RAGConfig,
    load_yaml, parse_config, load_config, merge_config,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Role", "ModelName", "TokenCount",
    "Temperature", "TopP", "MaxTokens",
    "Message", "GenerationParams",
    "ToolFunction", "ToolDefinition", "ToolCall",
    "NonEmptyList",
    "ChatRequest", "InferenceRequest", "InferenceResponse", "TokenUsage",
    # Result
    "Result", "Success", "Failure", "Railway",
    "map_result", "bind", "map_error", "tee",
    "unwrap_or", "unwrap_or_else", "validate_all",
    # Errors
    "ValidationError", "TokenLimitError",
    "ModelNotFoundError", "GenerationError",
    "InferenceError", "error_to_dict",
    # Validation
    "validate_messages", "validate_params", "check_token_limit",
    # Pipeline
    "pipe", "compose", "identity", "const", "curry2", "flip",
    # Config
    "ServerConfig", "EmbeddingServerConfig",
    "ModelDefinition", "ModelsConfig",
    "EmbeddingModelDefinition", "EmbeddingModelsConfig",
    "InferenceConfig", "AppConfig",
    "EmbeddingConfig", "RAGConfig",
    "load_yaml", "parse_config", "load_config", "merge_config",
]
