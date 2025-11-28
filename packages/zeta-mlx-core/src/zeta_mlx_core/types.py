"""도메인 타입 정의 (불변, 검증됨)"""
from dataclasses import dataclass
from typing import Literal, NewType, Self, Generic, TypeVar

T = TypeVar('T')

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

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(0.7)

    @classmethod
    def create(cls, value: float) -> 'Result[Temperature, str]':
        """검증 후 생성"""
        from zeta_mlx_core.result import Success, Failure
        if not 0.0 <= value <= 2.0:
            return Failure(f"Temperature must be 0.0-2.0, got {value}")
        return Success(cls(value))


@dataclass(frozen=True)
class TopP:
    """0.0 ~ 1.0 범위의 Top-P값"""
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"TopP must be 0.0-1.0, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(0.9)


@dataclass(frozen=True)
class MaxTokens:
    """1 ~ 32768 범위의 최대 토큰"""
    value: int

    def __post_init__(self) -> None:
        if not 1 <= self.value <= 32768:
            raise ValueError(f"MaxTokens must be 1-32768, got {self.value}")

    @classmethod
    def default(cls) -> Self:
        return cls(2048)

    @classmethod
    def create(cls, value: int) -> 'Result[MaxTokens, str]':
        """검증 후 생성"""
        from zeta_mlx_core.result import Success, Failure
        if not 1 <= value <= 32768:
            return Failure(f"MaxTokens must be 1-32768, got {value}")
        return Success(cls(value))


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
    parameters: dict


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
class NonEmptyList(Generic[T]):
    """최소 1개 이상의 요소를 가진 리스트"""
    head: T
    tail: tuple[T, ...]

    @classmethod
    def of(cls, items: list[T]) -> 'Result[NonEmptyList[T], str]':
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
class ChatRequest:
    """간단한 채팅 요청 (CLI/직접 사용용)"""
    model: str
    messages: list[Message]
    params: GenerationParams
    stream: bool = False


@dataclass(frozen=True)
class InferenceRequest:
    """추론 요청 (검증됨)"""
    model: ModelName
    messages: 'NonEmptyList[Message]'
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


# Forward reference resolution
from zeta_mlx_core.result import Result  # noqa: E402
