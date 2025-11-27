# Railway Oriented Programming (ROP) in Python

Scott Wlaschin의 Railway Oriented Programming을 Python/MLX 환경에 적용한 에러 처리 패턴입니다.

## 철도 비유

```
성공 트랙:  ────[검증]────[처리]────[변환]────> Success(결과)
                 ↘         ↘         ↘
실패 트랙:  ──────────────────────────────────> Failure(에러)
```

각 함수는 "철도 스위치"처럼 동작:
- 성공 → 다음 단계로 진행
- 실패 → 실패 트랙으로 분기 (나머지 단계 건너뜀)

## Python 구현

### Result 타입 정의

```python
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')

@dataclass(frozen=True)
class Success(Generic[T]):
    value: T

@dataclass(frozen=True)
class Failure(Generic[E]):
    error: E

Result = Union[Success[T], Failure[E]]
```

### 핵심 연산자

```python
def bind(
    result: Result[T, E],
    fn: Callable[[T], Result[U, E]],
) -> Result[U, E]:
    """
    >>= (bind): 모나드 함수를 파이프라인에 연결
    성공이면 fn 실행, 실패면 그대로 전달
    """
    match result:
        case Success(value):
            return fn(value)
        case Failure() as f:
            return f

def map_result(
    result: Result[T, E],
    fn: Callable[[T], U],
) -> Result[U, E]:
    """
    map: 일반 함수를 Result 컨텍스트에서 실행
    """
    match result:
        case Success(value):
            return Success(fn(value))
        case Failure() as f:
            return f
```

## MLX 추론 파이프라인 적용

### 에러 타입 정의

```python
from enum import Enum, auto
from pydantic import BaseModel

class InferenceError(Enum):
    MODEL_NOT_LOADED = auto()
    TOKENIZATION_FAILED = auto()
    GENERATION_FAILED = auto()
    INVALID_INPUT = auto()
    CONTEXT_TOO_LONG = auto()

@dataclass(frozen=True)
class ErrorInfo:
    code: InferenceError
    message: str
    details: dict | None = None
```

### ROP 스타일 추론 파이프라인

```python
from typing import Iterator

def validate_request(
    request: ChatRequest,
) -> Result[ChatRequest, ErrorInfo]:
    """1단계: 요청 검증"""
    if not request.messages:
        return Failure(ErrorInfo(
            code=InferenceError.INVALID_INPUT,
            message="메시지가 비어있습니다",
        ))
    return Success(request)

def apply_template(
    request: ChatRequest,
    tokenizer,
) -> Result[str, ErrorInfo]:
    """2단계: 템플릿 적용"""
    try:
        messages = [{"role": m.role, "content": m.content}
                    for m in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return Success(prompt)
    except Exception as e:
        return Failure(ErrorInfo(
            code=InferenceError.TOKENIZATION_FAILED,
            message=str(e),
        ))

def check_context_length(
    prompt: str,
    tokenizer,
    max_context: int = 8192,
) -> Result[str, ErrorInfo]:
    """3단계: 컨텍스트 길이 검증"""
    token_count = len(tokenizer.encode(prompt))
    if token_count > max_context:
        return Failure(ErrorInfo(
            code=InferenceError.CONTEXT_TOO_LONG,
            message=f"컨텍스트 초과: {token_count} > {max_context}",
        ))
    return Success(prompt)

def generate_response(
    prompt: str,
    engine: MLXInferenceEngine,
    max_tokens: int,
) -> Result[str, ErrorInfo]:
    """4단계: 응답 생성"""
    try:
        response = engine.generate(prompt, max_tokens=max_tokens)
        return Success(response)
    except Exception as e:
        return Failure(ErrorInfo(
            code=InferenceError.GENERATION_FAILED,
            message=str(e),
        ))
```

### 파이프라인 합성

```python
def chat_completion_pipeline(
    request: ChatRequest,
    engine: MLXInferenceEngine,
) -> Result[ChatResponse, ErrorInfo]:
    """
    ROP 파이프라인: 각 단계가 실패하면 즉시 실패 트랙으로 분기
    """
    # 방법 1: 명시적 bind 체인
    result = validate_request(request)
    result = bind(result, lambda r: apply_template(r, engine.tokenizer))
    result = bind(result, lambda p: check_context_length(p, engine.tokenizer))
    result = bind(result, lambda p: generate_response(p, engine, request.max_tokens))
    result = map_result(result, lambda text: create_response(text, request))
    return result

    # 방법 2: 파이프 연산자 (라이브러리 사용시)
    # return (
    #     Success(request)
    #     >> validate_request
    #     >> partial(apply_template, tokenizer=engine.tokenizer)
    #     >> partial(check_context_length, tokenizer=engine.tokenizer)
    #     >> partial(generate_response, engine=engine)
    #     | create_response
    # )
```

## FastAPI 통합

### 엔드포인트에서 Result 처리

```python
from fastapi import HTTPException
from fastapi.responses import JSONResponse

def result_to_response(
    result: Result[ChatResponse, ErrorInfo],
) -> ChatResponse:
    """Result를 HTTP 응답으로 변환"""
    match result:
        case Success(response):
            return response
        case Failure(error):
            status_map = {
                InferenceError.INVALID_INPUT: 400,
                InferenceError.CONTEXT_TOO_LONG: 400,
                InferenceError.MODEL_NOT_LOADED: 503,
                InferenceError.TOKENIZATION_FAILED: 500,
                InferenceError.GENERATION_FAILED: 500,
            }
            raise HTTPException(
                status_code=status_map.get(error.code, 500),
                detail={"code": error.code.name, "message": error.message},
            )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    result = chat_completion_pipeline(request, engine)
    return result_to_response(result)
```

## returns 라이브러리 활용

```bash
poetry add returns
```

```python
from returns.result import Result, Success, Failure
from returns.pipeline import flow
from returns.pointfree import bind

def chat_pipeline(request: ChatRequest) -> Result[ChatResponse, ErrorInfo]:
    return flow(
        request,
        validate_request,
        bind(apply_template),
        bind(check_context_length),
        bind(generate_response),
        map_result(create_response),
    )
```

## 장점

| 특성 | 설명 |
|------|------|
| **명시적 에러** | 에러가 타입에 표현되어 누락 방지 |
| **합성 가능** | 작은 함수를 조합해 파이프라인 구축 |
| **테스트 용이** | 각 단계를 독립적으로 테스트 |
| **추적 가능** | 에러 발생 지점 명확 |

## 관련 문서

- [Functional Design](./functional-design.md) - 함수형 디자인 원칙
- [Type-Driven Design](./type-driven-design.md) - 타입 주도 설계
- [Composition Patterns](./composition-patterns.md) - 함수 합성 패턴
