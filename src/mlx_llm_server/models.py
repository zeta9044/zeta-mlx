"""Pydantic models for API requests and responses."""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    messages: list[Message]
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    model: Optional[str] = None  # OpenAI compatibility


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Chat completion choice."""

    index: int
    message: Message
    finish_reason: str


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


class StreamResponse(BaseModel):
    """OpenAI-compatible streaming response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    version: str


class ModelObject(BaseModel):
    """Model object in OpenAI format."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "organization-owner"


class ModelsResponse(BaseModel):
    """OpenAI-compatible models list response."""

    object: str = "list"
    data: List[ModelObject]
