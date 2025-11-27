# Type-Driven Design (타입 주도 설계)

Scott Wlaschin의 "Make Illegal States Unrepresentable" 원칙을 Python Pydantic과 타입 힌트로 구현합니다.

## 핵심 원칙

> "타입 시스템을 즉각적인 유닛 테스트로 활용하라"
> — Scott Wlaschin

잘못된 상태가 **표현 자체가 불가능**하도록 타입을 설계합니다.

## Pydantic 기반 타입 설계

### 원시 타입 래핑 (Primitive Obsession 방지)

```python
from pydantic import BaseModel, Field, field_validator
from typing import NewType, Literal

# 나쁜 예: 원시 타입 직접 사용
def process_user(name: str, email: str, age: int):
    pass  # name과 email을 혼동할 수 있음

# 좋은 예: 의미 있는 타입 정의
class UserName(BaseModel):
    value: str = Field(min_length=1, max_length=100)

class Email(BaseModel):
    value: str

    @field_validator('value')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('유효하지 않은 이메일')
        return v.lower()

class Age(BaseModel):
    value: int = Field(ge=0, le=150)
```

### MLX 서버 타입 설계

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

# 역할은 정해진 값만 허용 (Literal 타입)
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    """채팅 메시지 - 잘못된 역할 불가능"""
    role: Role  # "admin" 같은 잘못된 값 컴파일 타임 에러
    content: str

class ChatRequest(BaseModel):
    """채팅 요청 - 유효하지 않은 파라미터 불가능"""
    messages: list[Message]  # 빈 리스트 체크 가능
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
```

## 상태 머신을 타입으로 표현

### 추론 상태

```python
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class ModelNotLoaded:
    """모델이 로드되지 않은 상태"""
    model_name: str

@dataclass(frozen=True)
class ModelLoading:
    """모델 로딩 중"""
    model_name: str
    progress: float  # 0.0 ~ 1.0

@dataclass(frozen=True)
class ModelReady:
    """모델 준비 완료"""
    model_name: str
    model: "mx.Module"
    tokenizer: "Tokenizer"

@dataclass(frozen=True)
class ModelError:
    """모델 로딩 실패"""
    model_name: str
    error: str

# 유니온 타입으로 가능한 모든 상태 표현
ModelState = Union[ModelNotLoaded, ModelLoading, ModelReady, ModelError]

def handle_state(state: ModelState) -> str:
    """패턴 매칭으로 모든 상태 처리 강제"""
    match state:
        case ModelNotLoaded(name):
            return f"모델 {name} 로드 필요"
        case ModelLoading(name, progress):
            return f"로딩 중: {progress:.0%}"
        case ModelReady(name, _, _):
            return f"준비 완료: {name}"
        case ModelError(name, error):
            return f"에러: {error}"
```

### 스트리밍 응답 상태

```python
from pydantic import BaseModel
from typing import Literal, Union

class StreamStart(BaseModel):
    """스트림 시작 - role 할당"""
    type: Literal["start"] = "start"
    id: str
    role: Literal["assistant"]

class StreamContent(BaseModel):
    """콘텐츠 청크"""
    type: Literal["content"] = "content"
    content: str

class StreamEnd(BaseModel):
    """스트림 종료"""
    type: Literal["end"] = "end"
    finish_reason: Literal["stop", "length", "error"]

StreamChunk = Union[StreamStart, StreamContent, StreamEnd]

# 잘못된 순서 방지를 위한 타입 기반 상태 머신
class StreamState:
    def __init__(self):
        self._started = False
        self._ended = False

    def emit_start(self, id: str) -> StreamStart:
        if self._started:
            raise ValueError("이미 시작됨")
        self._started = True
        return StreamStart(id=id, role="assistant")

    def emit_content(self, content: str) -> StreamContent:
        if not self._started or self._ended:
            raise ValueError("잘못된 상태")
        return StreamContent(content=content)

    def emit_end(self, reason: str = "stop") -> StreamEnd:
        if not self._started or self._ended:
            raise ValueError("잘못된 상태")
        self._ended = True
        return StreamEnd(finish_reason=reason)
```

## 검증 레이어

### Pydantic Validator로 비즈니스 규칙 인코딩

```python
from pydantic import BaseModel, Field, model_validator
from typing import Self

class InferenceConfig(BaseModel):
    """추론 설정 - 상호 의존적 검증"""
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=1)
    do_sample: bool = True

    @model_validator(mode='after')
    def validate_sampling(self) -> Self:
        """샘플링 비활성화시 temperature는 0이어야 함"""
        if not self.do_sample and self.temperature != 0.0:
            raise ValueError(
                "do_sample=False일 때 temperature는 0.0이어야 합니다"
            )
        return self

class ModelConfig(BaseModel):
    """모델 설정 - 조건부 필수 필드"""
    name: str
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    adapter_path: str | None = None

    @model_validator(mode='after')
    def validate_adapter(self) -> Self:
        """quantization이 none일 때만 adapter 사용 가능"""
        if self.adapter_path and self.quantization != "none":
            raise ValueError(
                "어댑터는 양자화되지 않은 모델에서만 사용 가능"
            )
        return self
```

## Discriminated Union (태그된 유니온)

```python
from pydantic import BaseModel, Field
from typing import Literal, Union, Annotated

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"

class AudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    audio_url: str
    format: Literal["mp3", "wav"]

# Discriminated Union - type 필드로 구분
Content = Annotated[
    Union[TextContent, ImageContent, AudioContent],
    Field(discriminator="type")
]

class MultiModalMessage(BaseModel):
    role: Role
    content: list[Content]

# 사용 예
msg = MultiModalMessage(
    role="user",
    content=[
        {"type": "text", "text": "이 이미지를 설명해줘"},
        {"type": "image", "image_url": "https://..."},
    ]
)
```

## Generic 타입으로 재사용

```python
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    """제네릭 API 응답"""
    success: bool
    data: T | None = None
    error: str | None = None

# 구체화된 타입
ChatApiResponse = ApiResponse[ChatResponse]
ModelsApiResponse = ApiResponse[list[ModelObject]]

# 사용
def create_success(data: T) -> ApiResponse[T]:
    return ApiResponse(success=True, data=data)

def create_error(error: str) -> ApiResponse[None]:
    return ApiResponse(success=False, error=error)
```

## 타입 체커 설정

### pyproject.toml

```toml
[tool.mypy]
python_version = "3.12"
strict = true
plugins = ["pydantic.mypy"]

[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
```

### 타입 체크 실행

```bash
poetry add --group dev mypy
poetry run mypy src/
```

## 장점 요약

| 원칙 | Python 구현 | 효과 |
|------|-------------|------|
| 잘못된 상태 불가능 | Literal, Union | 런타임 에러 방지 |
| 원시 타입 래핑 | Pydantic Model | 의미 있는 타입 |
| 상태 머신 | Union + match | 모든 케이스 처리 강제 |
| 비즈니스 규칙 | Validator | 컴파일 타임 검증 |

## 관련 문서

- [Functional Design](./functional-design.md) - 함수형 디자인 원칙
- [Railway Oriented Programming](./railway-oriented-programming.md) - 에러 처리 패턴
- [Composition Patterns](./composition-patterns.md) - 함수 합성 패턴
