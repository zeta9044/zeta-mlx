# Domain Modeling Made Functional

Scott Wlaschin의 "Domain Modeling Made Functional"을 Python에 적용합니다.

> 참고: [Domain Modeling Made Functional - Pragmatic Bookshelf](https://pragprog.com/titles/swdddf/domain-modeling-made-functional/)

## 핵심 원칙

### 1. Make Illegal States Unrepresentable

타입 시스템을 활용하여 잘못된 상태를 컴파일 타임에 방지합니다.

```python
from dataclasses import dataclass
from typing import Literal, NewType

# BAD: 잘못된 상태 가능
@dataclass
class BadMessage:
    role: str  # 아무 문자열이나 가능
    content: str

# GOOD: 잘못된 상태 불가능
Role = Literal["system", "user", "assistant", "tool"]

@dataclass(frozen=True)
class Message:
    role: Role  # 지정된 값만 가능
    content: str
```

### 2. Types as Documentation

타입 자체가 문서가 됩니다.

```python
from typing import NewType

# 의미 있는 타입 정의
ModelName = NewType('ModelName', str)
Temperature = NewType('Temperature', float)
TokenCount = NewType('TokenCount', int)

# 함수 시그니처가 문서화됨
def generate(
    model: ModelName,
    temperature: Temperature,
    max_tokens: TokenCount
) -> str:
    ...

# 호출 시 의도가 명확
generate(
    model=ModelName("Qwen3-8B"),
    temperature=Temperature(0.7),
    max_tokens=TokenCount(2048)
)
```

### 3. Constrained Types

값의 범위를 타입으로 제한합니다.

```python
from dataclasses import dataclass
from typing import Self

@dataclass(frozen=True)
class Temperature:
    """0.0 ~ 2.0 범위의 온도"""
    value: float

    def __post_init__(self):
        if not 0.0 <= self.value <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {self.value}")

    @classmethod
    def create(cls, value: float) -> 'Result[Self, str]':
        """검증된 Temperature 생성"""
        if 0.0 <= value <= 2.0:
            return Success(cls(value))
        return Failure(f"Temperature must be 0.0-2.0, got {value}")

@dataclass(frozen=True)
class NonEmptyList[T]:
    """최소 1개 이상의 요소를 가진 리스트"""
    head: T
    tail: list[T]

    @classmethod
    def create(cls, items: list[T]) -> 'Result[Self, str]':
        if not items:
            return Failure("List cannot be empty")
        return Success(cls(head=items[0], tail=items[1:]))

    def to_list(self) -> list[T]:
        return [self.head] + self.tail

    def __len__(self) -> int:
        return 1 + len(self.tail)
```

## AND 타입 vs OR 타입

### AND 타입 (Product Type)

모든 필드가 필요합니다.

```python
from dataclasses import dataclass

# AND 타입: role AND content 둘 다 필요
@dataclass(frozen=True)
class Message:
    role: Role
    content: str

# AND 타입: 모든 필드 필요
@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: list[str]
```

### OR 타입 (Sum Type / Discriminated Union)

여러 가능성 중 하나입니다.

```python
from dataclasses import dataclass
from typing import Union

# OR 타입: Success OR Failure
@dataclass(frozen=True)
class Success[T]:
    value: T

@dataclass(frozen=True)
class Failure[E]:
    error: E

Result = Success[T] | Failure[E]

# OR 타입: 여러 에러 종류
@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str

@dataclass(frozen=True)
class NetworkError:
    url: str
    status_code: int

@dataclass(frozen=True)
class ModelError:
    model: str
    reason: str

# InferenceError = ValidationError | NetworkError | ModelError
InferenceError = Union[ValidationError, NetworkError, ModelError]
```

### 패턴 매칭으로 OR 타입 처리

```python
def handle_error(error: InferenceError) -> str:
    match error:
        case ValidationError(field, message):
            return f"Validation failed for {field}: {message}"
        case NetworkError(url, status_code):
            return f"Network error {status_code} for {url}"
        case ModelError(model, reason):
            return f"Model {model} failed: {reason}"
```

## Bounded Context

시스템을 독립적인 컨텍스트로 분리합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Zeta MLX Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Inference   │  │     RAG      │  │  LangChain   │       │
│  │   Context    │  │   Context    │  │   Context    │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │ - Message    │  │ - Document   │  │ - Tool       │       │
│  │ - Generation │  │ - Chunk      │  │ - ToolCall   │       │
│  │ - Response   │  │ - Embedding  │  │ - Agent      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  각 Context는 자체 Ubiquitous Language를 가짐                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Context 간 통신

```python
# Inference Context의 타입
@dataclass(frozen=True)
class InferenceMessage:
    role: Literal["system", "user", "assistant"]
    content: str

# LangChain Context의 타입
@dataclass(frozen=True)
class LangChainMessage:
    type: Literal["human", "ai", "system"]
    content: str

# Anti-Corruption Layer: Context 간 변환
def to_langchain_message(msg: InferenceMessage) -> LangChainMessage:
    """Inference -> LangChain 변환"""
    type_map = {
        "user": "human",
        "assistant": "ai",
        "system": "system"
    }
    return LangChainMessage(
        type=type_map[msg.role],
        content=msg.content
    )

def from_langchain_message(msg: LangChainMessage) -> InferenceMessage:
    """LangChain -> Inference 변환"""
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system"
    }
    return InferenceMessage(
        role=role_map[msg.type],
        content=msg.content
    )
```

## Workflow as Pipeline

비즈니스 로직을 함수 파이프라인으로 표현합니다.

```python
from typing import Callable

# Workflow 타입 시그니처
InferenceWorkflow = Callable[
    [InferenceRequest],
    Result[InferenceResponse, InferenceError]
]

# 각 단계를 독립적인 함수로 정의
def create_inference_workflow(
    validate: Callable[[InferenceRequest], Result[InferenceRequest, InferenceError]],
    check_limits: Callable[[InferenceRequest], Result[InferenceRequest, InferenceError]],
    generate: Callable[[InferenceRequest], Result[InferenceResponse, InferenceError]],
    log_result: Callable[[InferenceResponse], None]
) -> InferenceWorkflow:
    """워크플로우 생성 (의존성 주입)"""

    def workflow(request: InferenceRequest) -> Result[InferenceResponse, InferenceError]:
        return (
            Railway.of(request)
            .bind(validate)
            .bind(check_limits)
            .bind(generate)
            .tee(log_result)
            .unwrap()
        )

    return workflow
```

## Onion Architecture

순수한 도메인 로직을 중심에 두고, I/O를 경계에 배치합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      I/O (Impure)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Application                         │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │              Domain (Pure)                   │    │    │
│  │  │                                              │    │    │
│  │  │   - Types (Message, Response)                │    │    │
│  │  │   - Business Rules                           │    │    │
│  │  │   - Workflows                                │    │    │
│  │  │                                              │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                                                      │    │
│  │   - Use Cases                                        │    │
│  │   - Orchestration                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│   - HTTP API (FastAPI)                                       │
│   - CLI (Click)                                              │
│   - Database                                                 │
│   - MLX Inference                                            │
└─────────────────────────────────────────────────────────────┘
```

### 계층별 코드 구조

```python
# ============================================================
# Domain Layer (Pure)
# ============================================================

@dataclass(frozen=True)
class Message:
    role: Role
    content: str

def validate_messages(messages: list[Message]) -> Result[list[Message], str]:
    """순수 함수: 메시지 검증"""
    if not messages:
        return Failure("Messages cannot be empty")
    return Success(messages)

# ============================================================
# Application Layer
# ============================================================

@dataclass(frozen=True)
class InferenceUseCase:
    """유스케이스: 의존성을 함수 파라미터로 받음"""
    generate_fn: Callable[[str], str]
    count_tokens_fn: Callable[[str], int]

    def execute(
        self,
        messages: list[Message],
        params: GenerationParams
    ) -> Result[str, InferenceError]:
        return (
            Railway.of(messages)
            .bind(validate_messages)
            .map(format_prompt)
            .bind(lambda p: self._check_tokens(p, params))
            .bind(lambda p: self._generate(p))
            .unwrap()
        )

    def _check_tokens(self, prompt: str, params: GenerationParams) -> Result[str, InferenceError]:
        count = self.count_tokens_fn(prompt)
        if count > params.max_tokens:
            return Failure(TokenLimitError(params.max_tokens, count))
        return Success(prompt)

    def _generate(self, prompt: str) -> Result[str, InferenceError]:
        try:
            return Success(self.generate_fn(prompt))
        except Exception as e:
            return Failure(ModelError("generation", str(e)))

# ============================================================
# Infrastructure Layer (Impure)
# ============================================================

def create_mlx_generator(model_name: str) -> Callable[[str], str]:
    """MLX 모델 생성기 (I/O)"""
    model, tokenizer = load_model(model_name)

    def generate(prompt: str) -> str:
        return mlx_generate(model, tokenizer, prompt)

    return generate
```

## DTO와 Domain Model 분리

외부 세계(JSON)와 내부 도메인을 분리합니다.

```python
from pydantic import BaseModel

# DTO (외부 세계 - 신뢰할 수 없음)
class ChatRequestDTO(BaseModel):
    """API 요청 DTO"""
    model: str
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None

# Domain Model (내부 - 검증됨)
@dataclass(frozen=True)
class ValidatedRequest:
    """검증된 도메인 모델"""
    model: ModelName
    messages: NonEmptyList[Message]
    params: GenerationParams

# 변환 함수 (Anti-Corruption Layer)
def to_domain(dto: ChatRequestDTO) -> Result[ValidatedRequest, ValidationError]:
    """DTO -> Domain Model 변환 (검증 포함)"""
    # 메시지 변환 및 검증
    messages_result = NonEmptyList.create([
        Message(role=m["role"], content=m["content"])
        for m in dto.messages
    ])

    match messages_result:
        case Failure(e):
            return Failure(ValidationError("messages", e))
        case Success(messages):
            return Success(ValidatedRequest(
                model=ModelName(dto.model),
                messages=messages,
                params=GenerationParams(
                    max_tokens=dto.max_tokens or 2048,
                    temperature=dto.temperature or 0.7,
                    top_p=0.9,
                    stop_sequences=[]
                )
            ))
```

## 참고 자료

- [Domain Modeling Made Functional](https://pragprog.com/titles/swdddf/domain-modeling-made-functional/)
- [Domain Modeling Made Functional: Takeaways](https://canro91.github.io/2021/12/13/DomainModelingMadeFunctional/)
- [F# for fun and profit - DDD](https://fsharpforfunandprofit.com/ddd/)
