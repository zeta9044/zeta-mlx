# API 패키지 (mlx-llm-api)

OpenAI 호환 HTTP API입니다. FastAPI 기반이며, DTO와 Domain을 명확히 분리합니다.

## 설계 원칙

- **DTO ≠ Domain**: 외부 JSON과 내부 타입 분리
- **Anti-Corruption Layer**: 변환 함수로 경계 보호
- **Result → HTTP Response**: Result 타입을 HTTP 상태로 변환
- **Streaming via SSE**: Server-Sent Events로 스트리밍

## 모듈 구조

```
mlx_llm_api/
├── __init__.py
├── app.py            # FastAPI 앱
├── routes/           # 라우트 핸들러
│   ├── __init__.py
│   ├── chat.py       # /v1/chat/completions
│   ├── models.py     # /v1/models
│   └── health.py     # /health
├── dto/              # Data Transfer Objects
│   ├── __init__.py
│   ├── requests.py   # 요청 DTO
│   └── responses.py  # 응답 DTO
├── converters.py     # DTO <-> Domain 변환
└── middleware.py     # CORS, 로깅
```

## dto/requests.py - 요청 DTO

```python
"""요청 DTO (외부 세계 - 신뢰할 수 없음)"""
from typing import Literal
from pydantic import BaseModel, Field


class MessageDTO(BaseModel):
    """메시지 DTO"""
    role: str  # 검증 전이므로 str
    content: str
    name: str | None = None


class ToolFunctionDTO(BaseModel):
    """도구 함수 DTO"""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class ToolDTO(BaseModel):
    """도구 DTO"""
    type: Literal["function"] = "function"
    function: ToolFunctionDTO


class ChatRequestDTO(BaseModel):
    """
    OpenAI Chat Completion 요청 DTO

    외부에서 들어오는 JSON 그대로입니다.
    검증은 Domain 변환 시 수행합니다.
    """
    model: str
    messages: list[MessageDTO]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    tools: list[ToolDTO] | None = None
    tool_choice: str | None = None


class TokenCountRequestDTO(BaseModel):
    """토큰 카운트 요청 DTO"""
    text: str
    model: str | None = None
```

## dto/responses.py - 응답 DTO

```python
"""응답 DTO (외부 세계로 전송)"""
from typing import Literal
from pydantic import BaseModel


class MessageResponseDTO(BaseModel):
    """메시지 응답 DTO"""
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[dict] | None = None


class ChoiceDTO(BaseModel):
    """선택지 DTO"""
    index: int
    message: MessageResponseDTO
    finish_reason: Literal["stop", "length", "tool_calls"] | None


class UsageDTO(BaseModel):
    """사용량 DTO"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseDTO(BaseModel):
    """
    OpenAI Chat Completion 응답 DTO

    OpenAI API 형식 그대로입니다.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChoiceDTO]
    usage: UsageDTO | None = None


# 스트리밍 응답
class DeltaDTO(BaseModel):
    """스트리밍 델타 DTO"""
    role: str | None = None
    content: str | None = None


class StreamChoiceDTO(BaseModel):
    """스트리밍 선택지 DTO"""
    index: int
    delta: DeltaDTO
    finish_reason: str | None = None


class StreamResponseDTO(BaseModel):
    """스트리밍 응답 DTO"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoiceDTO]


# 모델 목록
class ModelDTO(BaseModel):
    """모델 DTO"""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelsResponseDTO(BaseModel):
    """모델 목록 응답 DTO"""
    object: Literal["list"] = "list"
    data: list[ModelDTO]


# 에러 응답
class ErrorDetailDTO(BaseModel):
    """에러 상세 DTO"""
    message: str
    type: str
    code: str


class ErrorResponseDTO(BaseModel):
    """에러 응답 DTO"""
    error: ErrorDetailDTO


# 헬스체크
class HealthResponseDTO(BaseModel):
    """헬스체크 응답 DTO"""
    status: Literal["healthy", "unhealthy"]
    model: str | None
    version: str
```

## converters.py - DTO <-> Domain 변환

```python
"""
DTO <-> Domain 변환 (Anti-Corruption Layer)

외부 세계(DTO)와 내부 세계(Domain)를 분리합니다.
"""
from mlx_llm_core import (
    Result, Success, Failure,
    Message, GenerationParams, InferenceRequest, InferenceResponse,
    NonEmptyList, Temperature, TopP, MaxTokens, ModelName,
    ValidationError, InferenceError, error_to_dict,
)
from mlx_llm_api.dto.requests import ChatRequestDTO, MessageDTO
from mlx_llm_api.dto.responses import (
    ChatResponseDTO, ChoiceDTO, MessageResponseDTO, UsageDTO,
    ErrorResponseDTO, ErrorDetailDTO,
)


# ============================================================
# Request DTO → Domain
# ============================================================

def message_dto_to_domain(dto: MessageDTO) -> Result[Message, ValidationError]:
    """MessageDTO → Message"""
    valid_roles = {"system", "user", "assistant", "tool"}

    if dto.role not in valid_roles:
        return Failure(ValidationError(
            field="role",
            message=f"Invalid role: {dto.role}. Must be one of {valid_roles}"
        ))

    if not dto.content.strip():
        return Failure(ValidationError(
            field="content",
            message="Content cannot be empty"
        ))

    return Success(Message(
        role=dto.role,  # type: ignore (이미 검증됨)
        content=dto.content,
        name=dto.name,
    ))


def chat_request_to_domain(dto: ChatRequestDTO) -> Result[InferenceRequest, ValidationError]:
    """ChatRequestDTO → InferenceRequest"""
    # 메시지 변환
    messages: list[Message] = []
    for i, msg_dto in enumerate(dto.messages):
        result = message_dto_to_domain(msg_dto)
        match result:
            case Failure(err):
                return Failure(ValidationError(
                    field=f"messages[{i}]",
                    message=err.message
                ))
            case Success(msg):
                messages.append(msg)

    # NonEmptyList 생성
    messages_result = NonEmptyList.of(messages)
    match messages_result:
        case Failure(err):
            return Failure(ValidationError(field="messages", message=err))
        case Success(non_empty_messages):
            pass

    # 파라미터 생성 (기본값 적용)
    try:
        params = GenerationParams(
            max_tokens=MaxTokens(dto.max_tokens or 2048),
            temperature=Temperature(dto.temperature or 0.7),
            top_p=TopP(dto.top_p or 0.9),
            stop_sequences=tuple(dto.stop or []),
        )
    except ValueError as e:
        return Failure(ValidationError(field="params", message=str(e)))

    return Success(InferenceRequest(
        model=ModelName(dto.model),
        messages=non_empty_messages,
        params=params,
        stream=dto.stream,
    ))


# ============================================================
# Domain → Response DTO
# ============================================================

def inference_response_to_dto(
    response: InferenceResponse,
    request_id: str,
    model: str,
    created: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> ChatResponseDTO:
    """InferenceResponse → ChatResponseDTO"""
    return ChatResponseDTO(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChoiceDTO(
                index=0,
                message=MessageResponseDTO(
                    role="assistant",
                    content=response.content,
                    tool_calls=None,  # TODO: Tool Calling 지원
                ),
                finish_reason=response.finish_reason,
            )
        ],
        usage=UsageDTO(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def error_to_response_dto(error: InferenceError) -> ErrorResponseDTO:
    """InferenceError → ErrorResponseDTO"""
    error_dict = error_to_dict(error)
    return ErrorResponseDTO(
        error=ErrorDetailDTO(
            message=error_dict.get("message", str(error)),
            type=type(error).__name__,
            code=error_dict.get("code", "UNKNOWN_ERROR"),
        )
    )


# ============================================================
# Result → HTTP Response
# ============================================================

def result_to_status_code(result: Result) -> int:
    """Result를 HTTP 상태 코드로 변환"""
    match result:
        case Success(_):
            return 200
        case Failure(err):
            match err:
                case ValidationError():
                    return 400
                case TokenLimitError():
                    return 400
                case ModelNotFoundError():
                    return 404
                case _:
                    return 500
```

## routes/chat.py - Chat Completions 라우트

```python
"""Chat Completions 라우트"""
import uuid
import time
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from mlx_llm_core import Success, Failure
from mlx_llm_inference import InferenceEngine
from mlx_llm_api.dto.requests import ChatRequestDTO
from mlx_llm_api.dto.responses import (
    ChatResponseDTO, StreamResponseDTO, StreamChoiceDTO, DeltaDTO,
    ErrorResponseDTO,
)
from mlx_llm_api.converters import (
    chat_request_to_domain,
    inference_response_to_dto,
    error_to_response_dto,
)


router = APIRouter(prefix="/v1", tags=["chat"])


def get_engine() -> InferenceEngine:
    """엔진 의존성 (실제로는 app.state에서)"""
    from mlx_llm_api.app import app
    return app.state.engine


@router.post("/chat/completions")
async def chat_completions(request: ChatRequestDTO):
    """
    OpenAI 호환 Chat Completions

    POST /v1/chat/completions
    """
    engine = get_engine()

    # DTO → Domain 변환 (검증 포함)
    domain_result = chat_request_to_domain(request)

    match domain_result:
        case Failure(err):
            raise HTTPException(
                status_code=400,
                detail=error_to_response_dto(err).model_dump(),
            )
        case Success(inference_request):
            pass

    # 요청 ID 생성
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_response(
                engine,
                inference_request,
                request_id,
                created,
            ),
            media_type="text/event-stream",
        )

    # 동기 생성
    result = engine.generate(
        inference_request.messages,
        inference_request.params,
    )

    match result:
        case Failure(err):
            raise HTTPException(
                status_code=500,
                detail=error_to_response_dto(err).model_dump(),
            )
        case Success(response):
            # 토큰 카운트
            prompt = engine._apply_template_fn(
                inference_request.messages.to_list()
            )
            prompt_tokens = engine.count_tokens(prompt)
            completion_tokens = engine.count_tokens(response.content)

            return inference_response_to_dto(
                response,
                request_id,
                inference_request.model,
                created,
                prompt_tokens,
                completion_tokens,
            )


async def _stream_response(
    engine: InferenceEngine,
    request: 'InferenceRequest',
    request_id: str,
    created: int,
) -> AsyncIterator[str]:
    """스트리밍 응답 Generator"""
    model = request.model

    # 첫 청크: role 전송
    first_chunk = StreamResponseDTO(
        id=request_id,
        created=created,
        model=model,
        choices=[
            StreamChoiceDTO(
                index=0,
                delta=DeltaDTO(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # 콘텐츠 스트리밍
    for chunk in engine.stream(request.messages, request.params):
        chunk_response = StreamResponseDTO(
            id=request_id,
            created=created,
            model=model,
            choices=[
                StreamChoiceDTO(
                    index=0,
                    delta=DeltaDTO(content=chunk),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk_response.model_dump_json()}\n\n"

    # 마지막 청크: finish_reason
    final_chunk = StreamResponseDTO(
        id=request_id,
        created=created,
        model=model,
        choices=[
            StreamChoiceDTO(
                index=0,
                delta=DeltaDTO(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```

## app.py - FastAPI 앱

```python
"""FastAPI 애플리케이션"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mlx_llm_core import AppConfig
from mlx_llm_inference import InferenceEngine
from mlx_llm_api.routes import chat, models, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클"""
    # Startup: 엔진 초기화
    config: AppConfig = app.state.config
    app.state.engine = InferenceEngine(
        model_name=config.model.name,
        max_context=config.model.context_length,
    )

    yield

    # Shutdown: 정리
    app.state.engine = None


def create_app(config: AppConfig | None = None) -> FastAPI:
    """앱 팩토리"""
    app = FastAPI(
        title="MLX LLM Server",
        description="OpenAI-compatible LLM server for Apple Silicon",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 설정 저장
    app.state.config = config or AppConfig()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우트 등록
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)

    return app


# 기본 앱 인스턴스
app = create_app()
```

## Public API (__init__.py)

```python
"""MLX LLM API - OpenAI Compatible HTTP API"""
from mlx_llm_api.app import create_app, app
from mlx_llm_api.dto.requests import ChatRequestDTO, TokenCountRequestDTO
from mlx_llm_api.dto.responses import (
    ChatResponseDTO, StreamResponseDTO,
    ModelsResponseDTO, HealthResponseDTO,
    ErrorResponseDTO,
)
from mlx_llm_api.converters import (
    chat_request_to_domain,
    inference_response_to_dto,
    error_to_response_dto,
)

__version__ = "0.1.0"

__all__ = [
    # App
    "create_app", "app",
    # DTOs
    "ChatRequestDTO", "TokenCountRequestDTO",
    "ChatResponseDTO", "StreamResponseDTO",
    "ModelsResponseDTO", "HealthResponseDTO",
    "ErrorResponseDTO",
    # Converters
    "chat_request_to_domain",
    "inference_response_to_dto",
    "error_to_response_dto",
]
```
