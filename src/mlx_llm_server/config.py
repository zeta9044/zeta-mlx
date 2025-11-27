"""Configuration for MLX LLM server."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server configuration settings."""

    model_name: str = "mlx-community/Qwen3-8B-4bit"
    host: str = "0.0.0.0"
    port: int = 9044
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


settings = Settings()
