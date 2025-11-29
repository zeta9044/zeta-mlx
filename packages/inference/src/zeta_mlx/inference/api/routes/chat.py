"""Chat Completion 라우트 (OpenAI 호환 Tool Calling 지원)"""
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from zeta_mlx.core import Failure, Success, ToolCall
from zeta_mlx.inference import ModelManager
from zeta_mlx.inference.engine import parse_tool_calls
from zeta_mlx.inference.api.dto.requests import ChatRequestDTO
from zeta_mlx.inference.api.dto.responses import ChatResponseDTO, ErrorResponseDTO
from zeta_mlx.inference.api.converters import (
    chat_request_dto_to_domain,
    create_chat_response,
    create_stream_chunk,
    create_error_response,
)

router = APIRouter(prefix="/v1", tags=["chat"])
_model_manager: ModelManager | None = None


def set_model_manager(manager: ModelManager) -> None:
    global _model_manager
    _model_manager = manager


def get_model_manager() -> ModelManager | None:
    return _model_manager


async def generate_stream(
    request: ChatRequestDTO,
    manager: ModelManager,
) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성기 (vLLM 호환 tool calling)"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
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
    has_tools = bool(domain_request.tools)

    try:
        if has_tools:
            # tools가 있으면 버퍼링 후 파싱 (vLLM 호환)
            collected_content = ""
            for chunk_text in engine.stream(domain_request):
                collected_content += chunk_text

            # tool_call 태그 파싱
            clean_content, tool_calls = parse_tool_calls(collected_content)

            if tool_calls:
                # tool_calls가 있으면 content는 null, tool_calls만 반환
                for i, tc in enumerate(tool_calls):
                    tool_chunk = create_stream_chunk(
                        content=None,
                        model=model_alias,
                        chunk_id=chunk_id,
                        tool_calls=[{
                            "index": i,
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function_name,
                                "arguments": tc.function_arguments,
                            }
                        }]
                    )
                    yield f"data: {tool_chunk.model_dump_json()}\n\n"

                final_chunk = create_stream_chunk(
                    content=None,
                    model=model_alias,
                    chunk_id=chunk_id,
                    finish_reason="tool_calls",
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
            else:
                # tool_calls가 없으면 clean_content 반환
                if clean_content:
                    content_chunk = create_stream_chunk(
                        content=clean_content,
                        model=model_alias,
                        chunk_id=chunk_id,
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

                final_chunk = create_stream_chunk(
                    content=None,
                    model=model_alias,
                    chunk_id=chunk_id,
                    finish_reason="stop",
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
        else:
            # tools가 없으면 일반 스트리밍
            for chunk_text in engine.stream(domain_request):
                chunk = create_stream_chunk(
                    content=chunk_text,
                    model=model_alias,
                    chunk_id=chunk_id,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

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
    """OpenAI 호환 Chat Completion 엔드포인트 (Tool Calling 지원)"""
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

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

    # 토큰 수 추정
    prompt_text = " ".join(m.content or "" for m in domain_request.messages)
    estimated_prompt_tokens = max(1, len(prompt_text) // 4)
    estimated_completion_tokens = max(1, len(generation_result.content or "") // 4)

    # Tool calls 변환: tool_calls가 있으면 content는 null
    tool_calls = list(generation_result.tool_calls) if generation_result.tool_calls else None
    content = None if tool_calls else generation_result.content

    return create_chat_response(
        content=content,
        model=model_alias,
        prompt_tokens=estimated_prompt_tokens,
        completion_tokens=estimated_completion_tokens,
        finish_reason=generation_result.finish_reason,
        tool_calls=tool_calls,
    )
