# Composition Patterns

Scott Wlaschin의 함수 합성 패턴을 Python에 적용합니다.

> 참고: [F# for fun and profit - Function Composition](https://fsharpforfunandprofit.com/posts/function-composition/)

## 기본 합성 패턴

### 1. 단순 합성 (Compose)

```
    f: A → B
    g: B → C
    ─────────────
    g ∘ f: A → C
```

```python
from typing import TypeVar, Callable

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

def compose(
    f: Callable[[A], B],
    g: Callable[[B], C]
) -> Callable[[A], C]:
    """f 실행 후 g 실행"""
    return lambda a: g(f(a))

# 사용
to_upper = lambda s: s.upper()
add_exclaim = lambda s: s + "!"

shout = compose(to_upper, add_exclaim)
result = shout("hello")  # "HELLO!"
```

### 2. 파이프라인 (Pipe)

왼쪽에서 오른쪽으로 읽기 쉬운 합성.

```python
from functools import reduce
from typing import Callable, Any

def pipe(*funcs: Callable) -> Callable:
    """여러 함수를 순차적으로 합성 (왼쪽 → 오른쪽)"""
    def apply(x: Any) -> Any:
        return reduce(lambda acc, f: f(acc), funcs, x)
    return apply

# 사용
process = pipe(
    str.strip,
    str.lower,
    lambda s: s.replace(" ", "_"),
    lambda s: f"mlx_{s}"
)

result = process("  Hello World  ")  # "mlx_hello_world"
```

### 3. Fluent Pipeline Builder

```python
from typing import TypeVar, Callable, Generic
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class Pipeline(Generic[T]):
    """Fluent 스타일 파이프라인"""
    value: T

    def map(self, f: Callable[[T], U]) -> 'Pipeline[U]':
        return Pipeline(f(self.value))

    def tap(self, f: Callable[[T], None]) -> 'Pipeline[T]':
        """부수효과 실행 후 원래 값 유지"""
        f(self.value)
        return self

    def filter(self, predicate: Callable[[T], bool], default: T) -> 'Pipeline[T]':
        """조건 검사"""
        if predicate(self.value):
            return self
        return Pipeline(default)

    def unwrap(self) -> T:
        return self.value

# 사용
result = (
    Pipeline("  Hello World  ")
    .map(str.strip)
    .map(str.lower)
    .tap(lambda s: print(f"Processing: {s}"))
    .map(lambda s: s.replace(" ", "_"))
    .unwrap()
)
# Processing: hello world
# result = "hello_world"
```

## Monadic 합성 패턴

### 1. Bind (flatMap) - Result 체이닝

```
    f: A → Result[B, E]
    g: B → Result[C, E]
    ─────────────────────
    f >=> g: A → Result[C, E]
```

```python
from typing import TypeVar, Callable

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
E = TypeVar('E')

def kleisli_compose(
    f: Callable[[T], Result[U, E]],
    g: Callable[[U], Result[V, E]]
) -> Callable[[T], Result[V, E]]:
    """Kleisli 합성: Result 반환 함수들을 합성"""
    def composed(x: T) -> Result[V, E]:
        result = f(x)
        match result:
            case Success(value):
                return g(value)
            case Failure() as err:
                return err
    return composed

# 사용
parse_int: Callable[[str], Result[int, str]] = ...
validate_positive: Callable[[int], Result[int, str]] = ...
double: Callable[[int], Result[int, str]] = ...

process = kleisli_compose(
    kleisli_compose(parse_int, validate_positive),
    double
)

result = process("42")  # Success(84)
```

### 2. Applicative - 병렬 검증

```python
from typing import TypeVar, Callable
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')

def lift2(
    f: Callable[[A, B], C],
    ra: Result[A, list[E]],
    rb: Result[B, list[E]]
) -> Result[C, list[E]]:
    """2개의 Result를 결합하여 함수 적용"""
    match (ra, rb):
        case (Success(a), Success(b)):
            return Success(f(a, b))
        case (Failure(e1), Failure(e2)):
            return Failure(e1 + e2)  # 에러 누적
        case (Failure(e), _) | (_, Failure(e)):
            return Failure(e)

def validate_all(*results: Result[T, E]) -> Result[list[T], list[E]]:
    """모든 검증 결과를 병렬로 수집"""
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

# 사용
@dataclass(frozen=True)
class UserInput:
    name: str
    email: str
    age: int

def validate_user(name: str, email: str, age: str) -> Result[UserInput, list[str]]:
    return lift2(
        lambda validated_list, _: UserInput(
            name=validated_list[0],
            email=validated_list[1],
            age=validated_list[2]
        ),
        validate_all(
            validate_name(name),
            validate_email(email),
            validate_age(age)
        ),
        Success(None)
    )
```

## 고차 함수 패턴

### 1. Partial Application

```python
from functools import partial

def generate(
    model: str,
    temperature: float,
    top_p: float,
    prompt: str
) -> str:
    return f"[{model}] {prompt}"

# 파라미터 고정
generate_qwen = partial(generate, "Qwen3-8B")
generate_creative = partial(generate_qwen, 0.9, 0.95)

# 최종 호출
result = generate_creative("Hello!")
```

### 2. Currying

```python
from typing import Callable

def curry2(f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    """2인자 함수를 커링"""
    return lambda a: lambda b: f(a, b)

def curry3(f: Callable[[A, B, C], D]) -> Callable[[A], Callable[[B], Callable[[C], D]]]:
    """3인자 함수를 커링"""
    return lambda a: lambda b: lambda c: f(a, b, c)

# 사용
def add(a: int, b: int) -> int:
    return a + b

curried_add = curry2(add)
add_5 = curried_add(5)
result = add_5(3)  # 8
```

### 3. Decorator Pattern (함수형)

```python
from typing import TypeVar, Callable
from functools import wraps
import time

F = TypeVar('F', bound=Callable)

def with_logging(f: F) -> F:
    """로깅 데코레이터"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f"Calling {f.__name__}")
        result = f(*args, **kwargs)
        print(f"Finished {f.__name__}")
        return result
    return wrapper

def with_timing(f: F) -> F:
    """타이밍 데코레이터"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{f.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

def with_retry(max_retries: int = 3) -> Callable[[F], F]:
    """재시도 데코레이터"""
    def decorator(f: F) -> F:
        @wraps(f)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Retry {attempt + 1}/{max_retries}")
        return wrapper
    return decorator

# 합성: 여러 데코레이터 조합
@with_logging
@with_timing
@with_retry(3)
def fetch_data(url: str) -> str:
    ...
```

## 의존성 주입 (함수형)

### Constructor Injection 대신 Parameter Injection

```python
from typing import Protocol, Callable
from dataclasses import dataclass

# 의존성 인터페이스
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class ModelGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...

# 의존성을 파라미터로 받는 함수
def create_inference_workflow(
    count_tokens: Callable[[str], int],
    generate: Callable[[str], str],
    max_context: int = 8192
) -> Callable[[str], Result[str, str]]:
    """워크플로우 팩토리: 의존성 주입"""

    def workflow(prompt: str) -> Result[str, str]:
        token_count = count_tokens(prompt)
        if token_count > max_context:
            return Failure(f"Token limit exceeded: {token_count} > {max_context}")

        try:
            return Success(generate(prompt))
        except Exception as e:
            return Failure(str(e))

    return workflow

# 사용: 의존성 주입
workflow = create_inference_workflow(
    count_tokens=lambda s: len(s.split()),
    generate=lambda s: f"Response to: {s}",
    max_context=1000
)

result = workflow("Hello world")
```

### Reader Monad 패턴

```python
from typing import TypeVar, Callable, Generic
from dataclasses import dataclass

Env = TypeVar('Env')
A = TypeVar('A')
B = TypeVar('B')

@dataclass
class Reader(Generic[Env, A]):
    """환경을 받아 값을 생성하는 계산"""
    run: Callable[[Env], A]

    def map(self, f: Callable[[A], B]) -> 'Reader[Env, B]':
        return Reader(lambda env: f(self.run(env)))

    def flat_map(self, f: Callable[[A], 'Reader[Env, B]']) -> 'Reader[Env, B]':
        return Reader(lambda env: f(self.run(env)).run(env))

    @staticmethod
    def ask() -> 'Reader[Env, Env]':
        """현재 환경 가져오기"""
        return Reader(lambda env: env)

# 환경 타입 정의
@dataclass(frozen=True)
class AppConfig:
    model_name: str
    max_tokens: int
    temperature: float

# Reader를 사용한 의존성 주입
def get_model_name() -> Reader[AppConfig, str]:
    return Reader.ask().map(lambda cfg: cfg.model_name)

def get_max_tokens() -> Reader[AppConfig, int]:
    return Reader.ask().map(lambda cfg: cfg.max_tokens)

# 사용
config = AppConfig(model_name="Qwen3-8B", max_tokens=2048, temperature=0.7)
model_name = get_model_name().run(config)  # "Qwen3-8B"
```

## Monoid를 활용한 합성

### 1. 문자열 결합

```python
@dataclass
class StringMonoid:
    @staticmethod
    def empty() -> str:
        return ""

    @staticmethod
    def combine(a: str, b: str) -> str:
        return a + b

# 여러 프롬프트 결합
prompts = ["System: You are helpful.", "User: Hello", "Assistant: "]
result = reduce(StringMonoid.combine, prompts, StringMonoid.empty())
```

### 2. 검증 결과 결합

```python
from typing import Callable

@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: list[str]

    @staticmethod
    def success() -> 'ValidationResult':
        return ValidationResult(True, [])

    @staticmethod
    def failure(error: str) -> 'ValidationResult':
        return ValidationResult(False, [error])

    def combine(self, other: 'ValidationResult') -> 'ValidationResult':
        return ValidationResult(
            self.is_valid and other.is_valid,
            self.errors + other.errors
        )

# 여러 검증을 Monoid로 결합
def validate_request(request: Request) -> ValidationResult:
    validations = [
        validate_messages(request),
        validate_params(request),
        validate_model(request),
    ]
    return reduce(
        lambda a, b: a.combine(b),
        validations,
        ValidationResult.success()
    )
```

### 3. Endomorphism Monoid

같은 타입을 반환하는 함수들의 합성.

```python
from typing import TypeVar, Callable

T = TypeVar('T')

# Endomorphism: T → T
Endo = Callable[[T], T]

def endo_combine(f: Endo[T], g: Endo[T]) -> Endo[T]:
    """두 endomorphism 합성"""
    return lambda x: g(f(x))

def endo_empty() -> Endo[T]:
    """항등 함수"""
    return lambda x: x

# 텍스트 전처리 파이프라인
preprocessors: list[Endo[str]] = [
    str.strip,
    str.lower,
    lambda s: s.replace("\n", " "),
    lambda s: " ".join(s.split()),  # 다중 공백 제거
]

preprocess = reduce(endo_combine, preprocessors, endo_empty())
result = preprocess("  Hello\n\n  World  ")  # "hello world"
```

## 참고 자료

- [F# for fun and profit - Composition](https://fsharpforfunandprofit.com/posts/function-composition/)
- [Functional Programming Design Patterns](https://www.slideshare.net/slideshow/fp-patterns-ndc-london2014/42373281)
- [Monoids in Practice](https://fsharpforfunandprofit.com/posts/monoids-without-tears/)
