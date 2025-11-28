"""DTOs for API"""
from zeta_mlx_inference.api.dto.requests import ChatRequestDTO, MessageDTO
from zeta_mlx_inference.api.dto.responses import (
    ChatResponseDTO, StreamResponseDTO,
    ChoiceDTO, MessageResponseDTO, UsageDTO,
    StreamChoiceDTO, DeltaDTO,
    ModelsResponseDTO, ModelDTO,
    HealthResponseDTO, ErrorResponseDTO,
)

__all__ = [
    "ChatRequestDTO", "MessageDTO",
    "ChatResponseDTO", "StreamResponseDTO",
    "ChoiceDTO", "MessageResponseDTO", "UsageDTO",
    "StreamChoiceDTO", "DeltaDTO",
    "ModelsResponseDTO", "ModelDTO",
    "HealthResponseDTO", "ErrorResponseDTO",
]
