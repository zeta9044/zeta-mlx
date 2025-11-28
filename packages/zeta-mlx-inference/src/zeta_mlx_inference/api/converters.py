"""DTO <-> Domain 변환기 (Anticorruption Layer)"""
import uuid
import time
from zeta_mlx_core import (
    Result, Success, Failure,
    Message, GenerationParams, ChatRequest,
    Temperature, TopP, MaxTokens,
    ValidationError,
)
from zeta_mlx_inference.api.dto.requests import ChatRequestDTO, MessageDTO
from zeta_mlx_inference.api.dto.responses import (
    ChatResponseDTO, ChoiceDTO, MessageResponseDTO, UsageDTO,
    StreamResponseDTO, StreamChoiceDTO, DeltaDTO,
    ModelDTO, ModelsResponseDTO,
    HealthResponseDTO, ErrorResponseDTO, ErrorDetailDTO,
)


# ============================================================
# Request DTO -> Domain (외부 -> 내부)
# ============================================================

def message_dto_to_domain(dto: MessageDTO) -> Message:
    """MessageDTO를 도메인 Message로 변환"""
    return Message(
        role=dto.role,
        content=dto.content,
        name=dto.name,
    )


def chat_request_dto_to_domain(dto: ChatRequestDTO) -> Result[ChatRequest, ValidationError]:
    """ChatRequestDTO를 도메인 ChatRequest로 변환 (검증 포함)"""
    try:
        # Temperature 검증
        temperature = None
        if dto.temperature is not None:
            temp_result = Temperature.create(dto.temperature)
            if isinstance(temp_result, Failure):
                return temp_result
            temperature = temp_result.value

        # TopP 검증
        top_p = None
        if dto.top_p is not None:
            top_p_result = TopP.create(dto.top_p)
            if isinstance(top_p_result, Failure):
                return top_p_result
            top_p = top_p_result.value

        # MaxTokens 검증
        max_tokens = None
        if dto.max_tokens is not None:
            max_tokens_result = MaxTokens.create(dto.max_tokens)
            if isinstance(max_tokens_result, Failure):
                return max_tokens_result
            max_tokens = max_tokens_result.value

        # Messages 변환
        messages = [message_dto_to_domain(m) for m in dto.messages]

        # GenerationParams 생성
        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=dto.stop,
        )

        # ChatRequest 생성
        request = ChatRequest(
            model=dto.model,
            messages=messages,
            params=params,
            stream=dto.stream,
        )

        return Success(request)

    except Exception as e:
        return Failure(ValidationError(
            field="request",
            message=str(e),
            value=None,
        ))


# ============================================================
# Domain -> Response DTO (내부 -> 외부)
# ============================================================

def create_chat_response(
    content: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
) -> ChatResponseDTO:
    """도메인 결과를 ChatResponseDTO로 변환"""
    return ChatResponseDTO(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChoiceDTO(
                index=0,
                message=MessageResponseDTO(
                    role="assistant",
                    content=content,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageDTO(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def create_stream_chunk(
    content: str | None,
    model: str,
    chunk_id: str,
    role: str | None = None,
    finish_reason: str | None = None,
) -> StreamResponseDTO:
    """스트리밍 청크 생성"""
    return StreamResponseDTO(
        id=chunk_id,
        created=int(time.time()),
        model=model,
        choices=[
            StreamChoiceDTO(
                index=0,
                delta=DeltaDTO(
                    role=role,
                    content=content,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def create_model_dto(model_id: str, created: int = 0) -> ModelDTO:
    """모델 정보 DTO 생성"""
    return ModelDTO(
        id=model_id,
        created=created,
        owned_by="zeta-mlx",
    )


def create_models_response(model_ids: list[str]) -> ModelsResponseDTO:
    """모델 목록 응답 생성"""
    return ModelsResponseDTO(
        data=[create_model_dto(mid) for mid in model_ids],
    )


def create_health_response(
    status: str,
    model: str | None,
    version: str,
) -> HealthResponseDTO:
    """헬스체크 응답 생성"""
    return HealthResponseDTO(
        status=status,
        model=model,
        version=version,
    )


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request",
) -> ErrorResponseDTO:
    """에러 응답 생성"""
    return ErrorResponseDTO(
        error=ErrorDetailDTO(
            message=message,
            type=error_type,
            code=code,
        )
    )
