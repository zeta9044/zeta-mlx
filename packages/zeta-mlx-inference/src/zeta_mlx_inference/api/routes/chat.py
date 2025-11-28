"""Chat Completion 라우트 (다중 모델 지원)"""
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from zeta_mlx_core import Failure, Success
from zeta_mlx_inference import ModelManager
from zeta_mlx_inference.api.dto.requests import ChatRequestDTO
from zeta_mlx_inference.api.dto.responses import ChatResponseDTO, ErrorResponseDTO
from zeta_mlx_inference.api.converters import (
    chat_request_dto_to_domain,
    create_chat_response,
    create_stream_chunk,
    create_error_response,
)

router = APIRouter(prefix="/v1", tags=["chat"])

# 모델 관리자 (의존성 주입으로 설정)
_model_manager: ModelManager | None = None


def set_model_manager(manager: ModelManager) -> None:
    """모델 관리자 설정"""
    global _model_manager
    _model_manager = manager


def get_model_manager() -> ModelManager | None:
    """모델 관리자 조회"""
    return _model_manager


async def generate_stream(
    request: ChatRequestDTO,
    manager: ModelManager,
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성기"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # 모델 별칭 해석
    model_alias = manager.resolve_alias(request.model)

    # 첫 번째 청크: role 설정
    first_chunk = create_stream_chunk(
        content=None,
        model=model_alias,
        chunk_id=chunk_id,
        role="assistant",
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # 도메인 요청으로 변환
    domain_result = chat_request_dto_to_domain(request)
    if isinstance(domain_result, Failure):
        error_chunk = create_stream_chunk(
            content=f"[Error: {domain_result.error.message}]",
            model=model_alias,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    domain_request = domain_result.value

    # 모델 엔진 가져오기
    engine_result = manager.get_engine(model_alias)
    if isinstance(engine_result, Failure):
        error_chunk = create_stream_chunk(
            content=f"[Error: Model '{model_alias}' not found]",
            model=model_alias,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    engine = engine_result.value

    # 스트리밍 생성
    try:
        for chunk_text in engine.stream(domain_request):
            chunk = create_stream_chunk(
                content=chunk_text,
                model=model_alias,
                chunk_id=chunk_id,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # 마지막 청크: finish_reason
        final_chunk = create_stream_chunk(
            content=None,
            model=model_alias,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = create_stream_chunk(
            content=f"[Error: {str(e)}]",
            model=model_alias,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


@router.post(
    "/chat/completions",
    response_model=ChatResponseDTO,
    responses={400: {"model": ErrorResponseDTO}},
)
async def chat_completions(request: ChatRequestDTO) -> ChatResponseDTO | StreamingResponse:
    """
    OpenAI 호환 Chat Completion 엔드포인트 (다중 모델)

    model 필드:
    - 빈 문자열 또는 생략: 기본 모델 사용
    - 모델 별칭: "qwen3-8b", "qwen2.5-7b" 등
    - HuggingFace 경로: "mlx-community/Qwen3-8B-4bit"
    """
    if _model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized",
        )

    # 모델 별칭 해석
    model_alias = _model_manager.resolve_alias(request.model)

    # 스트리밍 모드
    if request.stream:
        return StreamingResponse(
            generate_stream(request, _model_manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # 비스트리밍 모드
    domain_result = chat_request_dto_to_domain(request)
    if isinstance(domain_result, Failure):
        error = domain_result.error
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                message=error.message,
                error_type="validation_error",
                code=error.field,
            ).model_dump(),
        )

    domain_request = domain_result.value

    # 모델 엔진 가져오기
    engine_result = _model_manager.get_engine(model_alias)
    if isinstance(engine_result, Failure):
        raise HTTPException(
            status_code=404,
            detail=create_error_response(
                message=f"Model '{model_alias}' not found",
                error_type="model_not_found",
                code="model_not_found",
            ).model_dump(),
        )

    engine = engine_result.value

    # 추론 실행
    result = engine.generate(domain_request)

    if isinstance(result, Failure):
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message=str(result.error),
                error_type="inference_error",
                code="inference_failed",
            ).model_dump(),
        )

    generation_result = result.value

    return create_chat_response(
        content=generation_result.text,
        model=model_alias,
        prompt_tokens=generation_result.prompt_tokens,
        completion_tokens=generation_result.completion_tokens,
        finish_reason="stop",
    )
