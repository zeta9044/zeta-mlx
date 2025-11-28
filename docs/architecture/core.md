# Core 패키지 (zeta-mlx-core)

순수 도메인 레이어입니다. 외부 의존성이 없고 모든 함수가 순수합니다.

## 설계 원칙

- **Pure Functions Only**: 모든 함수는 부수효과 없음
- **Immutable Types**: 모든 타입은 `frozen=True`
- **Make Illegal States Unrepresentable**: 타입으로 제약 표현
- **No I/O**: 파일, 네트워크, MLX 의존성 없음

## 모듈 구조

```
zeta_mlx_core/
├── __init__.py       # Public API 노출
├── types.py          # 도메인 타입
├── result.py         # Result[T, E], Railway
├── errors.py         # 에러 타입
├── validation.py     # 순수 검증 함수
├── pipeline.py       # 합성 유틸리티
└── config.py         # 설정 타입
```

## types.py - 도메인 타입

```python
"""도메인 타입 정의 (불변, 검증됨)"""
from dataclasses import dataclass
from typing import Literal, NewType, Self

# ============================================================
# Constrained Types (값 제약)
# ============================================================

Role = Literal["system", "user", "assistant", "tool"]

# NewType으로 의미 부여
ModelName = NewType('ModelName', str)
TokenCount = NewType('TokenCount', int)


@dataclass(frozen=True)
class Temperature:
    """0.0 ~ 2.0 범위의 온도값"""
    value: float

    def __post_init__(self):
        if not 0.0 <= self.value <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {self.value}")

    @classmethod
    def of(cls, value: float) -> 'Result[Self, str]':
        """검증된 Temperature 생성"""
        from zeta_mlx_core.result import Success, Failure
        if 0.0 <= value <= 2.0:
            return Success(cls(value))
        return Failure(f"Temperature must be 0.0-2.0, got {value}")

    @classmethod
    def default(cls) -> Self:
        return cls(0.7)


@dataclass(frozen=True)
class TopP:
    """0.0 ~ 1.0 범위의 Top-P값"""
    value: float

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"TopP must be 0.0-1.0, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(0.9)


@dataclass(frozen=True)
class MaxTokens:
    """1 ~ 32768 범위의 최대 토큰"""
    value: int

    def __post_init__(self):
        if not 1 <= self.value <= 32768:
            raise ValueError(f"MaxTokens must be 1-32768, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(2048)


# ============================================================
# AND Types (Product Types)
# ============================================================

@dataclass(frozen=True)
class Message:
    """대화 메시지"""
    role: Role
    content: str
    name: str | None = None  # Tool 호출 시


@dataclass(frozen=True)
class GenerationParams:
    """생성 파라미터"""
    max_tokens: MaxTokens
    temperature: Temperature
    top_p: TopP
    stop_sequences: tuple[str, ...] = ()

    @classmethod
    def default(cls) -> Self:
        return cls(
            max_tokens=MaxTokens.default(),
            temperature=Temperature.default(),
            top_p=TopP.default(),
        )


@dataclass(frozen=True)
class ToolDefinition:
    """도구 정의"""
    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass(frozen=True)
class ToolCall:
    """도구 호출"""
    id: str
    name: str
    arguments: str  # JSON string


# ============================================================
# NonEmpty Collections
# ============================================================

@dataclass(frozen=True)
class NonEmptyList[T]:
    """최소 1개 이상의 요소를 가진 리스트"""
    head: T
    tail: tuple[T, ...]

    @classmethod
    def of(cls, items: list[T]) -> 'Result[Self, str]':
        """리스트에서 생성 (검증 포함)"""
        from zeta_mlx_core.result import Success, Failure
        if not items:
            return Failure("List cannot be empty")
        return Success(cls(head=items[0], tail=tuple(items[1:])))

    def to_list(self) -> list[T]:
        return [self.head] + list(self.tail)

    def __len__(self) -> int:
        return 1 + len(self.tail)

    def __iter__(self):
        yield self.head
        yield from self.tail


# ============================================================
# Request/Response Types
# ============================================================

@dataclass(frozen=True)
class InferenceRequest:
    """추론 요청 (검증됨)"""
    model: ModelName
    messages: NonEmptyList[Message]
    params: GenerationParams
    tools: tuple[ToolDefinition, ...] = ()
    stream: bool = False


@dataclass(frozen=True)
class InferenceResponse:
    """추론 응답"""
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


@dataclass(frozen=True)
class TokenUsage:
    """토큰 사용량"""
    prompt_tokens: TokenCount
    completion_tokens: TokenCount

    @property
    def total_tokens(self) -> TokenCount:
        return TokenCount(self.prompt_tokens + self.completion_tokens)
```

## result.py - Railway Oriented Programming

```python
"""Result 타입과 Railway 패턴"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')
E2 = TypeVar('E2')


# ============================================================
# Result Type (OR Type)
# ============================================================

@dataclass(frozen=True)
class Success(Generic[T]):
    """성공 트랙"""
    value: T

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@dataclass(frozen=True)
class Failure(Generic[E]):
    """실패 트랙"""
    error: E

    def __repr__(self) -> str:
        return f"Failure({self.error!r})"


Result = Union[Success[T], Failure[E]]


# ============================================================
# Result 연산 (순수 함수)
# ============================================================

def map_result(
    result: Result[T, E],
    f: Callable[[T], U]
) -> Result[U, E]:
    """Success 값에 함수 적용 (Functor)"""
    match result:
        case Success(value):
            return Success(f(value))
        case Failure() as err:
            return err


def bind(
    result: Result[T, E],
    f: Callable[[T], Result[U, E]]
) -> Result[U, E]:
    """Result 반환 함수 체이닝 (Monad)"""
    match result:
        case Success(value):
            return f(value)
        case Failure() as err:
            return err


def map_error(
    result: Result[T, E],
    f: Callable[[E], E2]
) -> Result[T, E2]:
    """Failure 에러 변환"""
    match result:
        case Success() as ok:
            return ok
        case Failure(error):
            return Failure(f(error))


def tee(
    result: Result[T, E],
    f: Callable[[T], None]
) -> Result[T, E]:
    """부수효과 실행 (로깅 등)"""
    match result:
        case Success(value):
            f(value)
    return result


def unwrap_or(result: Result[T, E], default: T) -> T:
    """값 추출 또는 기본값"""
    match result:
        case Success(value):
            return value
        case Failure():
            return default


def unwrap_or_else(result: Result[T, E], f: Callable[[E], T]) -> T:
    """값 추출 또는 에러로부터 계산"""
    match result:
        case Success(value):
            return value
        case Failure(error):
            return f(error)


# ============================================================
# Railway 파이프라인 빌더
# ============================================================

class Railway(Generic[T, E]):
    """Fluent Railway 파이프라인"""

    def __init__(self, result: Result[T, E]):
        self._result = result

    @classmethod
    def of(cls, value: T) -> 'Railway[T, E]':
        """Success로 시작"""
        return cls(Success(value))

    @classmethod
    def fail(cls, error: E) -> 'Railway[T, E]':
        """Failure로 시작"""
        return cls(Failure(error))

    @classmethod
    def from_result(cls, result: Result[T, E]) -> 'Railway[T, E]':
        """Result에서 생성"""
        return cls(result)

    def map(self, f: Callable[[T], U]) -> 'Railway[U, E]':
        """값 변환"""
        return Railway(map_result(self._result, f))

    def bind(self, f: Callable[[T], Result[U, E]]) -> 'Railway[U, E]':
        """Result 반환 함수 체이닝"""
        return Railway(bind(self._result, f))

    def tee(self, f: Callable[[T], None]) -> 'Railway[T, E]':
        """부수효과 실행"""
        return Railway(tee(self._result, f))

    def map_error(self, f: Callable[[E], E2]) -> 'Railway[T, E2]':
        """에러 변환"""
        return Railway(map_error(self._result, f))

    def recover(self, f: Callable[[E], Result[T, E]]) -> 'Railway[T, E]':
        """에러 복구 시도"""
        match self._result:
            case Success():
                return self
            case Failure(error):
                return Railway(f(error))

    def unwrap(self) -> Result[T, E]:
        """최종 Result 반환"""
        return self._result

    def unwrap_or(self, default: T) -> T:
        """값 또는 기본값"""
        return unwrap_or(self._result, default)

    def unwrap_or_raise(self, exception_fn: Callable[[E], Exception] = ValueError) -> T:
        """값 또는 예외"""
        match self._result:
            case Success(value):
                return value
            case Failure(error):
                raise exception_fn(str(error))


# ============================================================
# 병렬 검증 (Applicative)
# ============================================================

def validate_all(*results: Result[T, E]) -> Result[list[T], list[E]]:
    """모든 검증 실행, 에러 누적"""
    successes: list[T] = []
    failures: list[E] = []

    for r in results:
        match r:
            case Success(v):
                successes.append(v)
            case Failure(e):
                failures.append(e)

    if failures:
        return Failure(failures)
    return Success(successes)
```

## errors.py - 에러 타입

```python
"""에러 타입 정의 (OR Type)"""
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ValidationError:
    """검증 에러"""
    field: str
    message: str
    code: str = "VALIDATION_ERROR"


@dataclass(frozen=True)
class TokenLimitError:
    """토큰 제한 초과"""
    limit: int
    actual: int
    code: str = "TOKEN_LIMIT_EXCEEDED"


@dataclass(frozen=True)
class ModelNotFoundError:
    """모델 없음"""
    model: str
    code: str = "MODEL_NOT_FOUND"


@dataclass(frozen=True)
class GenerationError:
    """생성 에러"""
    reason: str
    code: str = "GENERATION_ERROR"


# OR Type: 모든 추론 에러
InferenceError = Union[
    ValidationError,
    TokenLimitError,
    ModelNotFoundError,
    GenerationError,
]


def error_to_dict(error: InferenceError) -> dict:
    """에러를 딕셔너리로 변환 (API 응답용)"""
    match error:
        case ValidationError(field, message, code):
            return {"code": code, "field": field, "message": message}
        case TokenLimitError(limit, actual, code):
            return {"code": code, "limit": limit, "actual": actual}
        case ModelNotFoundError(model, code):
            return {"code": code, "model": model}
        case GenerationError(reason, code):
            return {"code": code, "reason": reason}
```

## validation.py - 순수 검증 함수

```python
"""순수 검증 함수"""
from zeta_mlx_core.types import (
    Message, GenerationParams, InferenceRequest, NonEmptyList
)
from zeta_mlx_core.result import Result, Success, Failure
from zeta_mlx_core.errors import ValidationError, TokenLimitError


def validate_messages(
    messages: NonEmptyList[Message]
) -> Result[NonEmptyList[Message], ValidationError]:
    """메시지 검증"""
    # 첫 메시지가 assistant이면 안됨
    if messages.head.role == "assistant":
        return Failure(ValidationError(
            field="messages",
            message="First message cannot be from assistant"
        ))

    # 빈 content 검사
    for i, msg in enumerate(messages):
        if not msg.content.strip():
            return Failure(ValidationError(
                field=f"messages[{i}].content",
                message="Message content cannot be empty"
            ))

    return Success(messages)


def validate_params(
    params: GenerationParams
) -> Result[GenerationParams, ValidationError]:
    """파라미터 검증 (이미 타입에서 검증됨, 추가 비즈니스 규칙)"""
    # Temperature와 Top-P 조합 검증
    if params.temperature.value == 0 and params.top_p.value < 1.0:
        return Failure(ValidationError(
            field="temperature,top_p",
            message="When temperature is 0, top_p should be 1.0"
        ))

    return Success(params)


def check_token_limit(
    prompt_tokens: int,
    max_context: int,
    max_tokens: int
) -> Result[int, TokenLimitError]:
    """토큰 제한 검사"""
    total_needed = prompt_tokens + max_tokens
    if total_needed > max_context:
        return Failure(TokenLimitError(
            limit=max_context,
            actual=total_needed
        ))
    return Success(prompt_tokens)
```

## pipeline.py - 합성 유틸리티

```python
"""함수 합성 유틸리티"""
from typing import TypeVar, Callable
from functools import reduce

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def pipe(*funcs: Callable) -> Callable:
    """왼쪽에서 오른쪽으로 함수 합성"""
    def apply(x):
        return reduce(lambda acc, f: f(acc), funcs, x)
    return apply


def compose(*funcs: Callable) -> Callable:
    """오른쪽에서 왼쪽으로 함수 합성"""
    return pipe(*reversed(funcs))


def identity(x: A) -> A:
    """항등 함수"""
    return x


def const(value: A) -> Callable[[B], A]:
    """상수 함수"""
    return lambda _: value


def curry2(f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    """2인자 함수 커링"""
    return lambda a: lambda b: f(a, b)


def flip(f: Callable[[A, B], C]) -> Callable[[B, A], C]:
    """인자 순서 뒤집기"""
    return lambda b, a: f(a, b)
```

## config.py - 설정 타입 (YAML 지원)

```python
"""설정 타입 (Pydantic + YAML)"""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
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
    """다중 모델 설정"""
    default: str = "qwen3-8b"  # 기본 모델 별칭
    max_loaded: int = Field(default=2, ge=1, le=8)  # 동시 로드 최대 수
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
    설정 로드 (YAML + 환경변수 + 기본값)

    우선순위: YAML < 환경변수 < 기본값
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
```

## config.yaml 예시

```yaml
# Zeta MLX Server 설정 파일
# 위치: ./config.yaml 또는 ~/.config/zeta-mlx/config.yaml

server:
  host: 0.0.0.0
  port: 9044

models:
  default: qwen3-8b  # 기본 모델 별칭
  max_loaded: 2       # 동시 로드 최대 모델 수

  available:
    qwen3-8b:
      path: mlx-community/Qwen3-8B-4bit
      context_length: 8192
      quantization: 4bit
      description: "Qwen3 8B - 범용 대화"

    qwen3-4b:
      path: mlx-community/Qwen3-4B-4bit
      context_length: 8192
      quantization: 4bit
      description: "Qwen3 4B - 경량 빠른 응답"

    qwen2.5-7b:
      path: mlx-community/Qwen2.5-7B-Instruct-4bit
      context_length: 32768
      quantization: 4bit
      description: "Qwen2.5 7B - 긴 컨텍스트"

    llama3.2-3b:
      path: mlx-community/Llama-3.2-3B-Instruct-4bit
      context_length: 8192
      quantization: 4bit
      description: "Llama 3.2 3B - 경량"

inference:
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  stop_sequences: []
```

## Public API (__init__.py)

```python
"""Zeta MLX Core - Pure Domain Layer"""
from zeta_mlx_core.types import (
    # Constrained Types
    Role, ModelName, TokenCount,
    Temperature, TopP, MaxTokens,
    # Domain Types
    Message, GenerationParams,
    ToolDefinition, ToolCall,
    NonEmptyList,
    InferenceRequest, InferenceResponse, TokenUsage,
)
from zeta_mlx_core.result import (
    Result, Success, Failure,
    Railway,
    map_result, bind, map_error, tee,
    unwrap_or, unwrap_or_else,
    validate_all,
)
from zeta_mlx_core.errors import (
    ValidationError, TokenLimitError,
    ModelNotFoundError, GenerationError,
    InferenceError,
    error_to_dict,
)
from zeta_mlx_core.validation import (
    validate_messages, validate_params, check_token_limit,
)
from zeta_mlx_core.pipeline import (
    pipe, compose, identity, const, curry2, flip,
)
from zeta_mlx_core.config import (
    ServerConfig, ModelDefinition, ModelsConfig, InferenceConfig, AppConfig,
    load_yaml, parse_config, load_config, merge_config,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Role", "ModelName", "TokenCount",
    "Temperature", "TopP", "MaxTokens",
    "Message", "GenerationParams",
    "ToolDefinition", "ToolCall",
    "NonEmptyList",
    "InferenceRequest", "InferenceResponse", "TokenUsage",
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
    "ServerConfig", "ModelDefinition", "ModelsConfig", "InferenceConfig", "AppConfig",
    "load_yaml", "parse_config", "load_config", "merge_config",
]
```
