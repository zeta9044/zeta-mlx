"""설정 타입 (Pydantic + YAML)"""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
import yaml

from zeta_mlx_core.result import Result, Success, Failure
from zeta_mlx_core.errors import ValidationError


# ============================================================
# 서버 설정
# ============================================================

class ServerConfig(BaseModel):
    """서버 설정"""
    host: str = "0.0.0.0"
    port: int = Field(default=9044, ge=1, le=65535)

    model_config = {"frozen": True}


class EmbeddingServerConfig(BaseModel):
    """임베딩 서버 설정"""
    host: str = "0.0.0.0"
    port: int = Field(default=9045, ge=1, le=65535)

    model_config = {"frozen": True}


# ============================================================
# 모델 설정 (다중 모델 지원)
# ============================================================

class ModelDefinition(BaseModel):
    """개별 모델 정의"""
    path: str  # HuggingFace 경로
    context_length: int = Field(default=8192, ge=512)
    quantization: str = "4bit"
    description: str = ""

    model_config = {"frozen": True}


class ModelsConfig(BaseModel):
    """모델 설정"""
    default: str = "qwen3-8b"  # 기본 모델 별칭
    available: dict[str, ModelDefinition] = Field(default_factory=lambda: {
        "qwen3-8b": ModelDefinition(
            path="mlx-community/Qwen3-8B-4bit",
            context_length=8192,
            description="Qwen3 8B (4-bit quantized)",
        ),
    })

    model_config = {"frozen": True}

    def get_model(self, alias: str) -> ModelDefinition | None:
        """별칭으로 모델 정의 조회"""
        return self.available.get(alias)

    def get_default_model(self) -> ModelDefinition:
        """기본 모델 정의 반환"""
        return self.available[self.default]

    def list_aliases(self) -> list[str]:
        """사용 가능한 모델 별칭 목록"""
        return list(self.available.keys())


# ============================================================
# 임베딩 모델 설정 (다중 모델 지원)
# ============================================================

class EmbeddingModelDefinition(BaseModel):
    """개별 임베딩 모델 정의"""
    path: str  # HuggingFace 경로 또는 sentence-transformers 모델명
    provider: Literal["sentence-transformers", "mlx"] = "sentence-transformers"
    dimension: int = Field(default=384, ge=64, le=4096)
    description: str = ""

    model_config = {"frozen": True}


class EmbeddingModelsConfig(BaseModel):
    """임베딩 모델 설정"""
    default: str = "minilm"  # 기본 모델 별칭
    batch_size: int = Field(default=32, ge=1, le=256)
    available: dict[str, EmbeddingModelDefinition] = Field(default_factory=lambda: {
        "minilm": EmbeddingModelDefinition(
            path="all-MiniLM-L6-v2",
            provider="sentence-transformers",
            dimension=384,
            description="MiniLM L6 - 빠른 범용 임베딩",
        ),
    })

    model_config = {"frozen": True}

    def get_model(self, alias: str) -> EmbeddingModelDefinition | None:
        """별칭으로 모델 정의 조회"""
        return self.available.get(alias)

    def get_default_model(self) -> EmbeddingModelDefinition:
        """기본 모델 정의 반환"""
        return self.available[self.default]

    def list_aliases(self) -> list[str]:
        """사용 가능한 모델 별칭 목록"""
        return list(self.available.keys())


class EmbeddingConfig(BaseModel):
    """임베딩 설정 (RAG용 - 레거시 호환)"""
    provider: Literal["simple", "sentence-transformers"] = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = Field(default=384, ge=64, le=4096)
    batch_size: int = Field(default=32, ge=1, le=256)

    model_config = {"frozen": True}


# ============================================================
# RAG 설정
# ============================================================

class RAGConfig(BaseModel):
    """RAG 파이프라인 설정"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    system_template: str = """You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain relevant information, say so."""

    model_config = {"frozen": True}


# ============================================================
# 추론 설정
# ============================================================

class InferenceConfig(BaseModel):
    """추론 기본 설정"""
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop_sequences: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


# ============================================================
# 전체 앱 설정
# ============================================================

class AppConfig(BaseModel):
    """전체 애플리케이션 설정"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    embedding_server: EmbeddingServerConfig = Field(default_factory=EmbeddingServerConfig)
    embedding_models: EmbeddingModelsConfig = Field(default_factory=EmbeddingModelsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    model_config = {"frozen": True}


# ============================================================
# YAML 로더 (순수 함수)
# ============================================================

def load_yaml(path: Path) -> Result[dict, ValidationError]:
    """YAML 파일 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return Success(data or {})
    except FileNotFoundError:
        return Failure(ValidationError(
            field="config_path",
            message=f"Config file not found: {path}",
        ))
    except yaml.YAMLError as e:
        return Failure(ValidationError(
            field="config_yaml",
            message=f"Invalid YAML: {e}",
        ))


def parse_config(data: dict) -> Result[AppConfig, ValidationError]:
    """딕셔너리를 AppConfig로 파싱"""
    try:
        config = AppConfig(**data)
        return Success(config)
    except Exception as e:
        return Failure(ValidationError(
            field="config",
            message=str(e),
        ))


def load_config(path: Path | str | None = None) -> Result[AppConfig, ValidationError]:
    """
    설정 로드 (YAML + 기본값)

    우선순위: YAML < 기본값
    """
    if path is None:
        # 기본 경로들 탐색
        default_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path.home() / ".config" / "zeta-mlx" / "config.yaml",
        ]
        for p in default_paths:
            if p.exists():
                path = p
                break

    if path is None:
        # 설정 파일 없으면 기본값 사용
        return Success(AppConfig())

    path = Path(path)

    # YAML 로드 → 파싱
    yaml_result = load_yaml(path)
    if isinstance(yaml_result, Failure):
        return yaml_result

    return parse_config(yaml_result.value)


def merge_config(base: AppConfig, overrides: dict) -> AppConfig:
    """설정 병합 (CLI 인자 등)"""
    data = base.model_dump()

    # 중첩 딕셔너리 병합
    def deep_merge(d1: dict, d2: dict) -> dict:
        result = d1.copy()
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    merged = deep_merge(data, overrides)
    return AppConfig(**merged)
