"""요청 DTO (외부 세계 - 신뢰할 수 없음)"""
from typing import Literal
from pydantic import BaseModel, Field


class MessageDTO(BaseModel):
    """메시지 DTO"""
    role: str
    content: str
    name: str | None = None


class ToolFunctionDTO(BaseModel):
    """도구 함수 DTO"""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class ToolDTO(BaseModel):
    """도구 DTO"""
    type: Literal["function"] = "function"
    function: ToolFunctionDTO


class ChatRequestDTO(BaseModel):
    """OpenAI Chat Completion 요청 DTO"""
    model: str
    messages: list[MessageDTO]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    tools: list[ToolDTO] | None = None
    tool_choice: str | None = None


class TokenCountRequestDTO(BaseModel):
    """토큰 카운트 요청 DTO"""
    text: str
    model: str | None = None
