"""Chat Completion 라우트"""
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from mlx_llm_core import Failure
from mlx_llm_inference import InferenceEngine
from mlx_llm_api.dto.requests import ChatRequestDTO
from mlx_llm_api.dto.responses import ChatResponseDTO, ErrorResponseDTO
from mlx_llm_api.converters import (
    chat_request_dto_to_domain,
    create_chat_response,
    create_stream_chunk,
    create_error_response,
)

router = APIRouter(prefix="/v1", tags=["chat"])

# 추론 엔진 (의존성 주입으로 설정)
_engine: InferenceEngine | None = None


def set_inference_engine(engine: InferenceEngine) -> None:
    """추론 엔진 설정"""
    global _engine
    _engine = engine


def get_inference_engine() -> InferenceEngine | None:
    """추론 엔진 조회"""
    return _engine


async def generate_stream(
    request: ChatRequestDTO,
    engine: InferenceEngine,
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성기"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # 첫 번째 청크: role 설정
    first_chunk = create_stream_chunk(
        content=None,
        model=request.model,
        chunk_id=chunk_id,
        role="assistant",
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # 도메인 요청으로 변환
    domain_result = chat_request_dto_to_domain(request)
    if isinstance(domain_result, Failure):
        error_chunk = create_stream_chunk(
            content=f"[Error: {domain_result.error.message}]",
            model=request.model,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        return

    domain_request = domain_result.value

    # 스트리밍 생성
    try:
        for chunk_text in engine.stream(domain_request):
            chunk = create_stream_chunk(
                content=chunk_text,
                model=request.model,
                chunk_id=chunk_id,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # 마지막 청크: finish_reason
        final_chunk = create_stream_chunk(
            content=None,
            model=request.model,
            chunk_id=chunk_id,
            finish_reason="stop",
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = create_stream_chunk(
            content=f"[Error: {str(e)}]",
            model=request.model,
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
    """OpenAI 호환 Chat Completion 엔드포인트"""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized",
        )

    # 스트리밍 모드
    if request.stream:
        return StreamingResponse(
            generate_stream(request, _engine),
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

    # 추론 실행
    result = _engine.generate(domain_request)

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
        model=request.model,
        prompt_tokens=generation_result.prompt_tokens,
        completion_tokens=generation_result.completion_tokens,
        finish_reason="stop",
    )
