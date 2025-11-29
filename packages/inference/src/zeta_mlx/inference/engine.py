"""MLX 추론 엔진 (OpenAI 호환 Tool Calling 지원)"""
import json
import re
import uuid
from typing import Iterator, Callable

from zeta_mlx.core import (
    Result, Success, Failure, Railway,
    Message, GenerationParams, InferenceResponse, ChatRequest,
    ToolDefinition, ToolCall,
    GenerationError, NonEmptyList,
    check_token_limit,
)
from zeta_mlx.inference.loader import load_model, ModelBundle
from zeta_mlx.inference.streaming import mlx_stream_generator


GenerateFn = Callable[[str, GenerationParams], Result[str, GenerationError]]
StreamFn = Callable[[str, GenerationParams], Iterator[str]]
TokenCountFn = Callable[[str], int]
TemplateFn = Callable[[list[Message], tuple[ToolDefinition, ...]], str]


def create_mlx_generate(bundle: ModelBundle) -> GenerateFn:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    def mlx_generate(prompt: str, params: GenerationParams) -> Result[str, GenerationError]:
        try:
            sampler = make_sampler(
                temp=params.temperature.value,
                top_p=params.top_p.value,
            )
            response = generate(
                bundle.model,
                bundle.tokenizer,
                prompt=prompt,
                max_tokens=params.max_tokens.value,
                sampler=sampler,
                verbose=False,
            )
            return Success(response)
        except Exception as e:
            return Failure(GenerationError(reason=str(e)))

    return mlx_generate


def create_mlx_stream(bundle: ModelBundle) -> StreamFn:
    def mlx_stream(prompt: str, params: GenerationParams) -> Iterator[str]:
        yield from mlx_stream_generator(
            bundle.model,
            bundle.tokenizer,
            prompt,
            max_tokens=params.max_tokens.value,
            temperature=params.temperature.value,
            top_p=params.top_p.value,
        )

    return mlx_stream


def create_mlx_token_counter(bundle: ModelBundle) -> TokenCountFn:
    def count(text: str) -> int:
        return len(bundle.tokenizer.encode(text))
    return count


def _format_tools_for_prompt(tools: tuple[ToolDefinition, ...]) -> list[dict]:
    """OpenAI 형식의 tools를 Qwen3 chat template용 dict로 변환"""
    return [tool.to_dict() for tool in tools]


def create_mlx_template_applier(bundle: ModelBundle) -> TemplateFn:
    """MLX 템플릿 적용 함수 생성 (tools 지원)"""
    def apply_template(messages: list[Message], tools: tuple[ToolDefinition, ...] = ()) -> str:
        msg_dicts = []
        for m in messages:
            msg_dict = {"role": m.role, "content": m.content or ""}
            # assistant의 tool_calls 포함
            if m.tool_calls:
                msg_dict["tool_calls"] = [tc.to_dict() for tc in m.tool_calls]
            # tool 응답 메시지
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            msg_dicts.append(msg_dict)

        if hasattr(bundle.tokenizer, "apply_chat_template"):
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Qwen3는 tools 파라미터 지원
            if tools:
                kwargs["tools"] = _format_tools_for_prompt(tools)

            return bundle.tokenizer.apply_chat_template(msg_dicts, **kwargs)
        else:
            return _default_template(msg_dicts, tools)

    return apply_template


def _default_template(messages: list[dict], tools: tuple[ToolDefinition, ...] = ()) -> str:
    """기본 템플릿 (fallback)"""
    formatted = ""

    # Tools가 있으면 시스템 프롬프트에 추가
    if tools:
        tool_text = json.dumps([t.to_dict() for t in tools], ensure_ascii=False, indent=2)
        formatted += f"<|im_start|>system\nYou have access to the following tools:\n{tool_text}\n<|im_end|>\n"

    for msg in messages:
        role, content = msg["role"], msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def parse_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """모델 응답에서 tool calls 파싱 (Qwen3 형식)

    Qwen3는 다음 형식으로 tool call 반환:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "value1"}}
    </tool_call>

    Returns:
        (content_without_tool_calls, list_of_tool_calls)
    """
    tool_calls = []

    # Qwen3 tool_call 태그 파싱
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)
            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function_name=data.get("name", ""),
                function_arguments=json.dumps(data.get("arguments", {}), ensure_ascii=False),
            )
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue

    # tool_call 태그 제거한 content
    clean_content = re.sub(pattern, '', content, flags=re.DOTALL).strip()

    return clean_content, tool_calls


class InferenceEngine:
    """추론 엔진 (OpenAI 호환 Tool Calling 지원)"""

    def __init__(self, model_or_bundle: str | ModelBundle, max_context: int = 32768):
        if isinstance(model_or_bundle, str):
            self._bundle = load_model(model_or_bundle)
        else:
            self._bundle = model_or_bundle
        self._max_context = max_context

        self._generate_fn = create_mlx_generate(self._bundle)
        self._stream_fn = create_mlx_stream(self._bundle)
        self._count_tokens_fn = create_mlx_token_counter(self._bundle)
        self._apply_template_fn = create_mlx_template_applier(self._bundle)

    def generate(
        self,
        request_or_messages: ChatRequest | NonEmptyList[Message],
        params: GenerationParams | None = None,
    ) -> Result[InferenceResponse, GenerationError]:
        """동기 생성 (tool calling 지원)"""
        if isinstance(request_or_messages, ChatRequest):
            messages = request_or_messages.messages
            params = request_or_messages.params
            tools = request_or_messages.tools
        else:
            messages = request_or_messages.to_list()
            tools = ()
            if params is None:
                params = GenerationParams.default()

        # 템플릿 적용 (tools 포함)
        prompt = self._apply_template_fn(
            messages if isinstance(messages, list) else messages.to_list(),
            tools
        )

        # 토큰 검사
        prompt_tokens = self._count_tokens_fn(prompt)
        token_check = check_token_limit(
            prompt_tokens,
            self._max_context,
            params.max_tokens.value
        )

        match token_check:
            case Failure(err):
                return Failure(GenerationError(
                    reason=f"Token limit exceeded: {err.actual} > {err.limit}"
                ))
            case Success(_):
                pass

        # 생성
        result = self._generate_fn(prompt, params)

        if isinstance(result, Failure):
            return result

        content = result.value

        # Tool calls 파싱
        if tools:
            clean_content, tool_calls = parse_tool_calls(content)
            if tool_calls:
                return Success(InferenceResponse(
                    content=clean_content,
                    tool_calls=tuple(tool_calls),
                    finish_reason="tool_calls"
                ))

        return Success(InferenceResponse(
            content=content,
            finish_reason="stop"
        ))

    def stream(
        self,
        request_or_messages: ChatRequest | NonEmptyList[Message],
        params: GenerationParams | None = None,
    ) -> Iterator[str]:
        """스트리밍 생성"""
        if isinstance(request_or_messages, ChatRequest):
            messages = request_or_messages.messages
            params = request_or_messages.params
            tools = request_or_messages.tools
        else:
            messages = request_or_messages.to_list()
            tools = ()
            if params is None:
                params = GenerationParams.default()

        prompt = self._apply_template_fn(
            messages if isinstance(messages, list) else messages.to_list(),
            tools
        )
        yield from self._stream_fn(prompt, params)

    def count_tokens(self, text: str) -> int:
        return self._count_tokens_fn(text)

    def apply_template(self, messages: list[Message], tools: tuple[ToolDefinition, ...] = ()) -> str:
        return self._apply_template_fn(messages, tools)

    @property
    def model_name(self) -> str:
        return self._bundle.name
