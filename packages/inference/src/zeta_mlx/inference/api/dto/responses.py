"""응답 DTO (OpenAI 호환 Tool Calling 지원)"""
from typing import Literal
from pydantic import BaseModel


class FunctionCallDTO(BaseModel):
    """함수 호출 DTO"""
    name: str
    arguments: str


class ToolCallDTO(BaseModel):
    """도구 호출 DTO (OpenAI 호환)"""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCallDTO


class MessageResponseDTO(BaseModel):
    """메시지 응답 DTO"""
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCallDTO] | None = None


class ChoiceDTO(BaseModel):
    """선택지 DTO"""
    index: int
    message: MessageResponseDTO
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class UsageDTO(BaseModel):
    """사용량 DTO"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseDTO(BaseModel):
    """OpenAI Chat Completion 응답 DTO"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChoiceDTO]
    usage: UsageDTO | None = None


class ToolCallDeltaDTO(BaseModel):
    """스트리밍 도구 호출 델타 DTO"""
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: dict | None = None


class DeltaDTO(BaseModel):
    """스트리밍 델타 DTO"""
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDeltaDTO] | None = None


class StreamChoiceDTO(BaseModel):
    """스트리밍 선택지 DTO"""
    index: int
    delta: DeltaDTO
    finish_reason: str | None = None


class StreamResponseDTO(BaseModel):
    """스트리밍 응답 DTO"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoiceDTO]


class ModelDTO(BaseModel):
    """모델 DTO"""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsResponseDTO(BaseModel):
    """모델 목록 응답 DTO"""
    object: Literal["list"] = "list"
    data: list[ModelDTO]


class ErrorDetailDTO(BaseModel):
    """에러 상세 DTO"""
    message: str
    type: str
    code: str


class ErrorResponseDTO(BaseModel):
    """에러 응답 DTO"""
    error: ErrorDetailDTO


class HealthResponseDTO(BaseModel):
    """헬스체크 응답 DTO"""
    status: Literal["healthy", "unhealthy"]
    model: str | None = None
    version: str
