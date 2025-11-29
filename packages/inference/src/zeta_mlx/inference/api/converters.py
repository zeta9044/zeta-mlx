"""DTO <-> Domain 변환기 (OpenAI 호환 Tool Calling 지원)"""
import uuid
import time
from zeta_mlx.core import (
    Result, Success, Failure,
    Message, GenerationParams, ChatRequest,
    Temperature, TopP, MaxTokens,
    ToolFunction, ToolDefinition, ToolCall,
    ValidationError,
)
from zeta_mlx.inference.api.dto.requests import ChatRequestDTO, MessageDTO, ToolDTO
from zeta_mlx.inference.api.dto.responses import (
    ChatResponseDTO, ChoiceDTO, MessageResponseDTO, UsageDTO,
    StreamResponseDTO, StreamChoiceDTO, DeltaDTO,
    ModelDTO, ModelsResponseDTO,
    HealthResponseDTO, ErrorResponseDTO, ErrorDetailDTO,
    ToolCallDTO, FunctionCallDTO,
)


def message_dto_to_domain(dto: MessageDTO) -> Message:
    # assistant의 tool_calls 변환
    tool_calls = ()
    if dto.tool_calls:
        from zeta_mlx.core import ToolCall
        tool_calls = tuple(
            ToolCall(
                id=tc.id,
                type=tc.type,
                function_name=tc.function.name,
                function_arguments=tc.function.arguments,
            )
            for tc in dto.tool_calls
        )

    return Message(
        role=dto.role,
        content=dto.content,  # None 허용
        name=dto.name,
        tool_call_id=dto.tool_call_id,
        tool_calls=tool_calls,
    )


def tool_dto_to_domain(dto: ToolDTO) -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function=ToolFunction(
            name=dto.function.name,
            description=dto.function.description,
            parameters=dto.function.parameters,
        )
    )


def chat_request_dto_to_domain(dto: ChatRequestDTO) -> Result[ChatRequest, ValidationError]:
    try:
        temperature = None
        if dto.temperature is not None:
            temp_result = Temperature.create(dto.temperature)
            if isinstance(temp_result, Failure):
                return temp_result
            temperature = temp_result.value

        top_p = None
        if dto.top_p is not None:
            top_p_result = TopP.create(dto.top_p)
            if isinstance(top_p_result, Failure):
                return top_p_result
            top_p = top_p_result.value

        max_tokens = None
        if dto.max_tokens is not None:
            max_tokens_result = MaxTokens.create(dto.max_tokens)
            if isinstance(max_tokens_result, Failure):
                return max_tokens_result
            max_tokens = max_tokens_result.value

        messages = [message_dto_to_domain(m) for m in dto.messages]

        params = GenerationParams(
            temperature=temperature if temperature else Temperature.default(),
            top_p=top_p if top_p else TopP.default(),
            max_tokens=max_tokens if max_tokens else MaxTokens.default(),
            stop_sequences=tuple(dto.stop) if dto.stop else (),
        )

        tools = tuple(tool_dto_to_domain(t) for t in dto.tools) if dto.tools else ()

        request = ChatRequest(
            model=dto.model,
            messages=messages,
            params=params,
            stream=dto.stream,
            tools=tools,
            tool_choice=dto.tool_choice,
        )

        return Success(request)

    except Exception as e:
        return Failure(ValidationError(field="request", message=str(e)))


def create_chat_response(
    content: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    tool_calls: list[ToolCall] | None = None,
) -> ChatResponseDTO:
    tool_calls_dto = None
    if tool_calls:
        tool_calls_dto = [
            ToolCallDTO(
                id=tc.id,
                type=tc.type,
                function=FunctionCallDTO(
                    name=tc.function_name,
                    arguments=tc.function_arguments,
                )
            )
            for tc in tool_calls
        ]

    return ChatResponseDTO(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model,
        choices=[
            ChoiceDTO(
                index=0,
                message=MessageResponseDTO(
                    role="assistant",
                    content=content if not tool_calls else None,
                    tool_calls=tool_calls_dto,
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
    tool_calls: list[dict] | None = None,
) -> StreamResponseDTO:
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
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def create_model_dto(model_id: str, created: int = 0) -> ModelDTO:
    return ModelDTO(id=model_id, created=created, owned_by="zeta-mlx")


def create_models_response(model_ids: list[str]) -> ModelsResponseDTO:
    return ModelsResponseDTO(data=[create_model_dto(mid) for mid in model_ids])


def create_health_response(status: str, model: str | None, version: str) -> HealthResponseDTO:
    return HealthResponseDTO(status=status, model=model, version=version)


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "invalid_request",
) -> ErrorResponseDTO:
    return ErrorResponseDTO(
        error=ErrorDetailDTO(message=message, type=error_type, code=code)
    )
