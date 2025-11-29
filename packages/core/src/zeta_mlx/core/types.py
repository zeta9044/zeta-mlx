"""도메인 타입 정의 (불변, 검증됨)"""
from dataclasses import dataclass, field
from typing import Literal, NewType, Self, Generic, TypeVar
import json

T = TypeVar('T')

Role = Literal["system", "user", "assistant", "tool"]
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
        from zeta_mlx.core.result import Success, Failure
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

    @classmethod
    def create(cls, value: float) -> 'Result[TopP, str]':
        from zeta_mlx.core.result import Success, Failure
        if not 0.0 <= value <= 1.0:
            return Failure(f"TopP must be 0.0-1.0, got {value}")
        return Success(cls(value))


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
        from zeta_mlx.core.result import Success, Failure
        if not 1 <= value <= 32768:
            return Failure(f"MaxTokens must be 1-32768, got {value}")
        return Success(cls(value))


@dataclass(frozen=True)
class Message:
    """대화 메시지 (OpenAI 호환)"""
    role: Role
    content: str | None = None  # tool_call assistant message는 content가 null
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple['ToolCall', ...] = ()  # assistant의 tool calls


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
class ToolFunction:
    """도구 함수 정의"""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDefinition:
    """도구 정의 (OpenAI 호환)"""
    type: Literal["function"] = "function"
    function: ToolFunction = field(default_factory=lambda: ToolFunction("", "", {}))

    @classmethod
    def create(cls, name: str, description: str, parameters: dict | None = None) -> 'ToolDefinition':
        return cls(
            type="function",
            function=ToolFunction(
                name=name,
                description=description,
                parameters=parameters or {},
            )
        )

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "function": {
                "name": self.function.name,
                "description": self.function.description,
                "parameters": self.function.parameters,
            }
        }


@dataclass(frozen=True)
class ToolCall:
    """도구 호출 (OpenAI 호환)"""
    id: str
    type: Literal["function"] = "function"
    function_name: str = ""
    function_arguments: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function_name,
                "arguments": self.function_arguments,
            }
        }


@dataclass(frozen=True)
class NonEmptyList(Generic[T]):
    """최소 1개 이상의 요소를 가진 리스트"""
    head: T
    tail: tuple[T, ...]

    @classmethod
    def of(cls, items: list[T]) -> 'Result[NonEmptyList[T], str]':
        from zeta_mlx.core.result import Success, Failure
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


@dataclass(frozen=True)
class ChatRequest:
    """채팅 요청 (OpenAI 호환 tool calling 지원)"""
    model: str
    messages: list[Message]
    params: GenerationParams
    stream: bool = False
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | None = None


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


from zeta_mlx.core.result import Result
