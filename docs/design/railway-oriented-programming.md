# Railway Oriented Programming (ROP)

Scott Wlaschin의 Railway Oriented Programming을 Python에 적용합니다.

> 참고: [Railway Oriented Programming - F# for fun and profit](https://fsharpforfunandprofit.com/rop/)

## 핵심 개념: Two-Track Model

```
Success Track (Green)
═══════════════════════════════════════════════════►

Failure Track (Red)
═══════════════════════════════════════════════════►
```

모든 함수는 두 개의 트랙 중 하나로 결과를 보냅니다:
- **Success Track**: 정상 처리 계속
- **Failure Track**: 에러 발생 시 이동, 이후 함수들 스킵

## Result 타입 정의

```python
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union

T = TypeVar('T')  # Success 값 타입
E = TypeVar('E')  # Error 타입
U = TypeVar('U')  # 변환 후 타입

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

# Result 타입 = Success | Failure
Result = Union[Success[T], Failure[E]]
```

## 핵심 함수들

### 1. map - 일반 함수를 Result에 적용

```
    ┌─────────────────────────────────────────┐
    │           map(f)                        │
    │  Success ──► f(value) ──► Success       │
    │  Failure ──────────────► Failure        │
    └─────────────────────────────────────────┘
```

```python
def map(
    result: Result[T, E],
    f: Callable[[T], U]
) -> Result[U, E]:
    """일반 함수를 Success 값에 적용"""
    match result:
        case Success(value):
            return Success(f(value))
        case Failure() as err:
            return err
```

### 2. bind - Result 반환 함수를 체이닝

```
    ┌─────────────────────────────────────────┐
    │           bind(f)                       │
    │  Success ──► f(value) ──► Success/Fail  │
    │  Failure ──────────────► Failure        │
    └─────────────────────────────────────────┘
```

```python
def bind(
    result: Result[T, E],
    f: Callable[[T], Result[U, E]]
) -> Result[U, E]:
    """Result를 반환하는 함수를 체이닝 (flatMap)"""
    match result:
        case Success(value):
            return f(value)
        case Failure() as err:
            return err
```

### 3. map_error - 에러 변환

```python
def map_error(
    result: Result[T, E],
    f: Callable[[E], E2]
) -> Result[T, E2]:
    """Failure의 에러를 변환"""
    match result:
        case Success() as ok:
            return ok
        case Failure(error):
            return Failure(f(error))
```

### 4. tee - 부수효과 실행 (로깅 등)

```python
def tee(
    result: Result[T, E],
    f: Callable[[T], None]
) -> Result[T, E]:
    """부수효과 실행 후 원래 값 반환"""
    match result:
        case Success(value):
            f(value)
            return result
        case Failure():
            return result
```

### 5. try_catch - 예외를 Result로 변환

```python
def try_catch(
    f: Callable[[], T],
    error_handler: Callable[[Exception], E]
) -> Result[T, E]:
    """예외를 Result로 변환"""
    try:
        return Success(f())
    except Exception as e:
        return Failure(error_handler(e))
```

## Railway 파이프라인 빌더

```python
from typing import TypeVar, Callable, Generic
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')

class Railway(Generic[T, E]):
    """Railway 파이프라인 빌더"""

    def __init__(self, value: Result[T, E]):
        self._value = value

    @classmethod
    def of(cls, value: T) -> 'Railway[T, E]':
        """Success로 시작"""
        return cls(Success(value))

    @classmethod
    def fail(cls, error: E) -> 'Railway[T, E]':
        """Failure로 시작"""
        return cls(Failure(error))

    def map(self, f: Callable[[T], U]) -> 'Railway[U, E]':
        """일반 함수 적용"""
        return Railway(map(self._value, f))

    def bind(self, f: Callable[[T], Result[U, E]]) -> 'Railway[U, E]':
        """Result 반환 함수 체이닝"""
        return Railway(bind(self._value, f))

    def tee(self, f: Callable[[T], None]) -> 'Railway[T, E]':
        """부수효과 실행"""
        return Railway(tee(self._value, f))

    def map_error(self, f: Callable[[E], E2]) -> 'Railway[T, E2]':
        """에러 변환"""
        return Railway(map_error(self._value, f))

    def unwrap(self) -> Result[T, E]:
        """최종 Result 반환"""
        return self._value

    def unwrap_or(self, default: T) -> T:
        """Success면 값, Failure면 기본값"""
        match self._value:
            case Success(value):
                return value
            case Failure():
                return default

    def unwrap_or_raise(self) -> T:
        """Success면 값, Failure면 예외 발생"""
        match self._value:
            case Success(value):
                return value
            case Failure(error):
                raise ValueError(f"Railway failed: {error}")
```

## 실제 사용 예시: LLM 추론 파이프라인

```python
from dataclasses import dataclass
from typing import Literal

# 도메인 타입 정의
@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass(frozen=True)
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float

@dataclass(frozen=True)
class InferenceRequest:
    messages: list[Message]
    params: GenerationParams

@dataclass(frozen=True)
class InferenceResponse:
    content: str
    tokens_used: int

# 에러 타입 (OR 타입으로 정의)
@dataclass(frozen=True)
class ValidationError:
    field: str
    message: str

@dataclass(frozen=True)
class ModelError:
    code: str
    message: str

@dataclass(frozen=True)
class TokenLimitError:
    limit: int
    actual: int

InferenceError = ValidationError | ModelError | TokenLimitError

# 각 단계 함수 (Result 반환)
def validate_messages(
    request: InferenceRequest
) -> Result[InferenceRequest, InferenceError]:
    """메시지 검증"""
    if not request.messages:
        return Failure(ValidationError("messages", "메시지가 비어있습니다"))
    if request.messages[0].role == "assistant":
        return Failure(ValidationError("messages", "첫 메시지는 assistant일 수 없습니다"))
    return Success(request)

def validate_params(
    request: InferenceRequest
) -> Result[InferenceRequest, InferenceError]:
    """파라미터 검증"""
    if request.params.temperature < 0 or request.params.temperature > 2:
        return Failure(ValidationError("temperature", "0~2 범위여야 합니다"))
    return Success(request)

def check_token_limit(
    request: InferenceRequest,
    count_tokens: Callable[[str], int],
    max_context: int = 8192
) -> Result[InferenceRequest, InferenceError]:
    """토큰 제한 확인"""
    total = sum(count_tokens(m.content) for m in request.messages)
    if total > max_context:
        return Failure(TokenLimitError(max_context, total))
    return Success(request)

def generate_response(
    request: InferenceRequest,
    model_generate: Callable[[str], str]
) -> Result[InferenceResponse, InferenceError]:
    """응답 생성"""
    try:
        prompt = format_prompt(request.messages)
        content = model_generate(prompt)
        return Success(InferenceResponse(content=content, tokens_used=len(content)))
    except Exception as e:
        return Failure(ModelError("GENERATION_ERROR", str(e)))

def format_prompt(messages: list[Message]) -> str:
    """프롬프트 포맷팅"""
    return "\n".join(f"[{m.role}] {m.content}" for m in messages)

# Railway 파이프라인으로 조합
def inference_pipeline(
    request: InferenceRequest,
    count_tokens: Callable[[str], int],
    model_generate: Callable[[str], str]
) -> Result[InferenceResponse, InferenceError]:
    """추론 파이프라인"""
    return (
        Railway.of(request)
        .bind(validate_messages)
        .bind(validate_params)
        .bind(lambda r: check_token_limit(r, count_tokens))
        .tee(lambda r: print(f"Processing {len(r.messages)} messages"))
        .bind(lambda r: generate_response(r, model_generate))
        .unwrap()
    )
```

## 병렬 검증 (Applicative)

여러 검증을 동시에 실행하고 모든 에러를 수집합니다.

```python
def validate_all(
    *validations: Result[T, E]
) -> Result[list[T], list[E]]:
    """모든 검증 실행, 에러 수집"""
    successes = []
    failures = []

    for v in validations:
        match v:
            case Success(value):
                successes.append(value)
            case Failure(error):
                failures.append(error)

    if failures:
        return Failure(failures)
    return Success(successes)

# 사용
result = validate_all(
    validate_name(name),
    validate_email(email),
    validate_age(age),
)
# Failure([NameError(...), EmailError(...)]) - 모든 에러 수집
```

## When NOT to Use ROP

Scott Wlaschin의 경고:

> "This is a useful approach to error handling, but please don't take it to extremes!"

### ROP를 사용하지 말아야 할 경우

1. **간단한 코드**: 복잡성만 증가
2. **I/O 중심 코드**: async/await가 더 적합
3. **성능 크리티컬**: 오버헤드 발생
4. **팀이 익숙하지 않은 경우**: 학습 곡선

### ROP가 적합한 경우

1. **검증 파이프라인**: 여러 단계 검증
2. **비즈니스 로직**: 도메인 규칙 체이닝
3. **에러 수집이 필요한 경우**: 모든 에러 반환
4. **함수 합성이 자연스러운 경우**: 파이프라인 구조

## 참고 자료

- [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/)
- [Against Railway-Oriented Programming](https://fsharpforfunandprofit.com/posts/against-railway-oriented-programming/)
- [Chessie F# Library](https://github.com/fsprojects/Chessie)
