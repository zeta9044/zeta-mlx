"""MLX 추론 엔진"""
from typing import Iterator, Callable

from mlx_llm_core import (
    Result, Success, Failure, Railway,
    Message, GenerationParams, InferenceResponse,
    GenerationError, NonEmptyList,
    check_token_limit,
)
from mlx_llm_inference.loader import load_model, ModelBundle
from mlx_llm_inference.streaming import mlx_stream_generator


# ============================================================
# 타입 정의
# ============================================================

GenerateFn = Callable[[str, GenerationParams], Result[str, GenerationError]]
StreamFn = Callable[[str, GenerationParams], Iterator[str]]
TokenCountFn = Callable[[str], int]
TemplateFn = Callable[[list[Message]], str]


# ============================================================
# MLX 기반 구현 (Impure)
# ============================================================

def create_mlx_generate(bundle: ModelBundle) -> GenerateFn:
    """MLX generate 함수 생성"""
    from mlx_lm import generate

    def mlx_generate(prompt: str, params: GenerationParams) -> Result[str, GenerationError]:
        try:
            response = generate(
                bundle.model,
                bundle.tokenizer,
                prompt=prompt,
                max_tokens=params.max_tokens.value,
                temp=params.temperature.value,
                top_p=params.top_p.value,
                verbose=False,
            )
            return Success(response)
        except Exception as e:
            return Failure(GenerationError(reason=str(e)))

    return mlx_generate


def create_mlx_stream(bundle: ModelBundle) -> StreamFn:
    """MLX 스트리밍 함수 생성"""
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
    """MLX 토큰 카운터 생성"""
    def count(text: str) -> int:
        return len(bundle.tokenizer.encode(text))
    return count


def create_mlx_template_applier(bundle: ModelBundle) -> TemplateFn:
    """MLX 템플릿 적용 함수 생성"""
    def apply_template(messages: list[Message]) -> str:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        if hasattr(bundle.tokenizer, "apply_chat_template"):
            return bundle.tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return _default_template(msg_dicts)

    return apply_template


def _default_template(messages: list[dict]) -> str:
    """기본 템플릿 (fallback)"""
    formatted = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


# ============================================================
# 추론 엔진
# ============================================================

class InferenceEngine:
    """
    추론 엔진 (Facade)

    MLX 모델을 로드하고 추론 워크플로우를 제공합니다.
    """

    def __init__(self, model_name: str, max_context: int = 8192):
        self._bundle = load_model(model_name)
        self._max_context = max_context

        # 의존성 생성
        self._generate_fn = create_mlx_generate(self._bundle)
        self._stream_fn = create_mlx_stream(self._bundle)
        self._count_tokens_fn = create_mlx_token_counter(self._bundle)
        self._apply_template_fn = create_mlx_template_applier(self._bundle)

    def generate(
        self,
        messages: NonEmptyList[Message],
        params: GenerationParams,
    ) -> Result[InferenceResponse, GenerationError]:
        """동기 생성"""
        prompt = self._apply_template_fn(messages.to_list())

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

        return (
            Railway.from_result(result)
            .map(lambda content: InferenceResponse(
                content=content,
                finish_reason="stop"
            ))
            .unwrap()
        )

    def stream(
        self,
        messages: NonEmptyList[Message],
        params: GenerationParams,
    ) -> Iterator[str]:
        """스트리밍 생성"""
        prompt = self._apply_template_fn(messages.to_list())
        yield from self._stream_fn(prompt, params)

    def count_tokens(self, text: str) -> int:
        """토큰 카운트"""
        return self._count_tokens_fn(text)

    def apply_template(self, messages: list[Message]) -> str:
        """템플릿 적용"""
        return self._apply_template_fn(messages)

    @property
    def model_name(self) -> str:
        return self._bundle.name
