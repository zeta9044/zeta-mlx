# Composition Patterns (함수 합성 패턴)

Scott Wlaschin의 "작은 함수를 조합해 큰 기능 구축" 원칙을 Python/MLX에 적용합니다.

## 핵심 개념

```
작은 함수들:  [f] [g] [h]
               ↓   ↓   ↓
합성된 함수:  [f >> g >> h] = 하나의 파이프라인
```

## Python 함수 합성 기법

### 1. functools를 활용한 합성

```python
from functools import reduce, partial
from typing import Callable, TypeVar

T = TypeVar('T')

def compose(*functions: Callable) -> Callable:
    """오른쪽에서 왼쪽으로 함수 합성 (수학적 합성)"""
    return reduce(
        lambda f, g: lambda x: f(g(x)),
        functions,
        lambda x: x
    )

def pipe(*functions: Callable) -> Callable:
    """왼쪽에서 오른쪽으로 함수 합성 (파이프라인)"""
    return reduce(
        lambda f, g: lambda x: g(f(x)),
        functions,
        lambda x: x
    )

# 사용 예
normalize = lambda s: s.strip().lower()
tokenize = lambda s: s.split()
count = lambda tokens: len(tokens)

# compose: count(tokenize(normalize(text)))
word_count = compose(count, tokenize, normalize)

# pipe: text -> normalize -> tokenize -> count
word_count = pipe(normalize, tokenize, count)

print(word_count("  Hello World  "))  # 2
```

### 2. partial을 활용한 함수 특화

```python
from functools import partial

def generate(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """범용 생성 함수"""
    ...

# 특화된 함수 생성
generate_creative = partial(generate, temperature=1.2, max_tokens=4096)
generate_precise = partial(generate, temperature=0.1, max_tokens=1024)

# 모델 바인딩
generate_with_qwen = partial(generate, model=qwen_model, tokenizer=qwen_tokenizer)
```

### 3. Decorator를 활용한 합성

```python
from functools import wraps
from typing import Callable, ParamSpec, TypeVar
import time

P = ParamSpec('P')
R = TypeVar('R')

def with_timing(fn: Callable[P, R]) -> Callable[P, R]:
    """실행 시간 측정 데코레이터"""
    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{fn.__name__}: {elapsed:.3f}s")
        return result
    return wrapper

def with_retry(max_attempts: int = 3):
    """재시도 데코레이터"""
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"재시도 {attempt + 1}/{max_attempts}")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def with_validation(validator: Callable[[dict], bool]):
    """입력 검증 데코레이터"""
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not validator(kwargs):
                raise ValueError("검증 실패")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

# 합성하여 사용
@with_timing
@with_retry(max_attempts=3)
def generate_response(prompt: str, model) -> str:
    return model.generate(prompt)
```

## MLX 추론 파이프라인

### Generator 합성

```python
from typing import Iterator, Callable

def create_token_stream(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int,
) -> Iterator[int]:
    """토큰 스트림 생성"""
    tokens = mx.array(tokenizer.encode(prompt))
    for token, _ in zip(generate_step(tokens, model), range(max_tokens)):
        if token == tokenizer.eos_token_id:
            return
        yield token

def decode_tokens(
    token_stream: Iterator[int],
    tokenizer,
) -> Iterator[str]:
    """토큰을 텍스트로 디코딩"""
    buffer = []
    prev_text = ""
    for token in token_stream:
        buffer.append(token)
        text = tokenizer.decode(buffer)
        new_text = text[len(prev_text):]
        if new_text and '\ufffd' not in new_text:
            yield new_text
            prev_text = text

def format_sse(
    text_stream: Iterator[str],
    completion_id: str,
) -> Iterator[str]:
    """SSE 형식으로 포맷팅"""
    for chunk in text_stream:
        data = {"id": completion_id, "content": chunk}
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"

# 파이프라인 합성
def create_streaming_pipeline(
    model,
    tokenizer,
    max_tokens: int = 2048,
) -> Callable[[str], Iterator[str]]:
    """스트리밍 파이프라인 팩토리"""

    def pipeline(prompt: str) -> Iterator[str]:
        token_stream = create_token_stream(prompt, model, tokenizer, max_tokens)
        text_stream = decode_tokens(token_stream, tokenizer)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return format_sse(text_stream, completion_id)

    return pipeline
```

### 고차 함수로 미들웨어 패턴

```python
from typing import Callable, Awaitable
from fastapi import Request, Response

Middleware = Callable[
    [Request, Callable[[Request], Awaitable[Response]]],
    Awaitable[Response]
]

def logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Awaitable[Response]:
    """로깅 미들웨어"""
    print(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    return response

def rate_limit_middleware(
    requests_per_minute: int,
) -> Middleware:
    """속도 제한 미들웨어 팩토리"""
    limiter = RateLimiter(requests_per_minute)

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not limiter.allow():
            return Response(status_code=429)
        return await call_next(request)

    return middleware
```

## 함수형 데이터 변환

### map, filter, reduce 패턴

```python
from functools import reduce
from typing import Iterable, Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

# map: 각 요소 변환
def transform_messages(
    messages: list[dict],
    transformer: Callable[[dict], dict],
) -> list[dict]:
    return list(map(transformer, messages))

# filter: 조건에 맞는 요소만 선택
def filter_messages(
    messages: list[dict],
    predicate: Callable[[dict], bool],
) -> list[dict]:
    return list(filter(predicate, messages))

# reduce: 누적 연산
def concatenate_content(messages: list[dict]) -> str:
    return reduce(
        lambda acc, msg: acc + msg["content"] + "\n",
        messages,
        ""
    )

# 합성하여 사용
pipeline = pipe(
    partial(filter_messages, predicate=lambda m: m["role"] != "system"),
    partial(transform_messages, transformer=lambda m: {**m, "content": m["content"].strip()}),
    concatenate_content,
)

result = pipeline(messages)
```

### Lens 패턴 (불변 업데이트)

```python
from typing import TypeVar, Callable, Any
from copy import deepcopy

T = TypeVar('T')

def lens_get(path: list[str]) -> Callable[[dict], Any]:
    """중첩된 딕셔너리에서 값 읽기"""
    def getter(obj: dict) -> Any:
        for key in path:
            obj = obj[key]
        return obj
    return getter

def lens_set(path: list[str], value: Any) -> Callable[[dict], dict]:
    """불변 업데이트"""
    def setter(obj: dict) -> dict:
        result = deepcopy(obj)
        target = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return result
    return setter

# 사용 예
config = {
    "model": {
        "name": "qwen3",
        "params": {"temperature": 0.7}
    }
}

get_temperature = lens_get(["model", "params", "temperature"])
set_temperature = lens_set(["model", "params", "temperature"], 1.0)

print(get_temperature(config))  # 0.7
new_config = set_temperature(config)  # 불변 업데이트
print(get_temperature(new_config))  # 1.0
```

## 커스텀 파이프 연산자

```python
from typing import TypeVar, Callable, Generic

T = TypeVar('T')
U = TypeVar('U')

class Pipe(Generic[T]):
    """파이프 연산자를 지원하는 래퍼"""

    def __init__(self, value: T):
        self._value = value

    def __or__(self, fn: Callable[[T], U]) -> "Pipe[U]":
        """| 연산자로 함수 적용"""
        return Pipe(fn(self._value))

    def __rshift__(self, fn: Callable[[T], U]) -> "Pipe[U]":
        """>> 연산자로 함수 적용"""
        return Pipe(fn(self._value))

    @property
    def value(self) -> T:
        return self._value

# 사용 예
result = (
    Pipe("  Hello World  ")
    | str.strip
    | str.lower
    | str.split
    | len
).value

print(result)  # 2

# 또는 >> 연산자
result = (
    Pipe(messages)
    >> validate_messages
    >> apply_template
    >> generate_response
).value
```

## 정리

| 패턴 | 용도 | Python 구현 |
|------|------|-------------|
| pipe/compose | 함수 체이닝 | `reduce`, 커스텀 클래스 |
| partial | 함수 특화 | `functools.partial` |
| decorator | 관심사 분리 | `@decorator` |
| map/filter/reduce | 데이터 변환 | 내장 함수, Generator |
| lens | 불변 업데이트 | 커스텀 헬퍼 |

## 관련 문서

- [Functional Design](./functional-design.md) - 함수형 디자인 원칙
- [Railway Oriented Programming](./railway-oriented-programming.md) - 에러 처리 패턴
- [Type-Driven Design](./type-driven-design.md) - 타입 주도 설계
