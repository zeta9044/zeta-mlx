"""설정 타입 (Pydantic)"""
from typing import Literal
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """서버 설정"""
    host: str = "127.0.0.1"
    port: int = Field(default=9044, ge=1, le=65535)

    model_config = {"frozen": True}


class ModelConfig(BaseModel):
    """모델 설정"""
    name: str = "mlx-community/Qwen3-8B-4bit"
    context_length: int = Field(default=8192, ge=512)

    model_config = {"frozen": True}


class InferenceConfig(BaseModel):
    """추론 설정"""
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class AppConfig(BaseModel):
    """전체 설정"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    model_config = {"frozen": True}
