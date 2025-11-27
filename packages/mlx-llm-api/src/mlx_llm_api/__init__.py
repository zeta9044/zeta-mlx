"""MLX LLM API - FastAPI Server"""
from mlx_llm_api.app import create_app, create_app_with_engine
from mlx_llm_api.dto import (
    ChatRequestDTO, MessageDTO,
    ChatResponseDTO, StreamResponseDTO,
    ChoiceDTO, MessageResponseDTO, UsageDTO,
    StreamChoiceDTO, DeltaDTO,
    ModelsResponseDTO, ModelDTO,
    HealthResponseDTO, ErrorResponseDTO,
)
from mlx_llm_api.converters import (
    chat_request_dto_to_domain,
    create_chat_response,
    create_stream_chunk,
    create_models_response,
    create_health_response,
    create_error_response,
)

__version__ = "0.1.0"

__all__ = [
    # App factory
    "create_app",
    "create_app_with_engine",
    # Request DTOs
    "ChatRequestDTO",
    "MessageDTO",
    # Response DTOs
    "ChatResponseDTO",
    "StreamResponseDTO",
    "ChoiceDTO",
    "MessageResponseDTO",
    "UsageDTO",
    "StreamChoiceDTO",
    "DeltaDTO",
    "ModelsResponseDTO",
    "ModelDTO",
    "HealthResponseDTO",
    "ErrorResponseDTO",
    # Converters
    "chat_request_dto_to_domain",
    "create_chat_response",
    "create_stream_chunk",
    "create_models_response",
    "create_health_response",
    "create_error_response",
]
