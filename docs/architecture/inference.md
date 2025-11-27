# Inference 패키지 (mlx-llm-inference)

MLX 기반 추론 엔진입니다. I/O 경계에 위치하며, Core의 순수 타입을 사용합니다.

## 설계 원칙

- **I/O at the Edge**: MLX 호출은 이 레이어에서만
- **Return Result**: 모든 함수는 `Result[T, E]` 반환
- **Dependency Injection**: 의존성은 함수 파라미터로 주입
- **Generator for Streaming**: 스트리밍은 Generator 패턴

## 모듈 구조

```
mlx_llm_inference/
├── __init__.py       # Public API
├── engine.py         # 추론 엔진 (메인 진입점)
├── loader.py         # 모델 로더
├── streaming.py      # 스트리밍 Generator
├── tokenizer.py      # 토큰 카운터
├── registry.py       # 모델 레지스트리
└── custom_models/    # 커스텀 모델
    ├── __init__.py
    └── qwen3.py
```

## engine.py - 추론 엔진

```python
"""MLX 추론 엔진"""
from typing import Iterator, Callable
from functools import partial

from mlx_llm_core import (
    Result, Success, Failure, Railway,
    Message, GenerationParams, InferenceResponse,
    GenerationError, TokenLimitError,
    validate_messages, check_token_limit,
    NonEmptyList,
)
from mlx_llm_inference.loader import load_model, ModelBundle
from mlx_llm_inference.streaming import create_stream_generator
from mlx_llm_inference.tokenizer import count_tokens, apply_chat_template


# ============================================================
# 타입 정의
# ============================================================

# 함수 타입 (의존성 주입용)
GenerateFn = Callable[[str, GenerationParams], Result[str, GenerationError]]
StreamFn = Callable[[str, GenerationParams], Iterator[str]]
TokenCountFn = Callable[[str], int]
TemplateFn = Callable[[list[Message]], str]


# ============================================================
# 추론 워크플로우 (순수 함수 + I/O 주입)
# ============================================================

def create_inference_workflow(
    generate_fn: GenerateFn,
    count_tokens_fn: TokenCountFn,
    apply_template_fn: TemplateFn,
    max_context: int = 8192,
) -> Callable[[NonEmptyList[Message], GenerationParams], Result[InferenceResponse, GenerationError]]:
    """
    추론 워크플로우 팩토리

    의존성을 주입받아 워크플로우 함수를 반환합니다.
    """

    def workflow(
        messages: NonEmptyList[Message],
        params: GenerationParams,
    ) -> Result[InferenceResponse, GenerationError]:
        # 프롬프트 생성 (순수)
        prompt = apply_template_fn(messages.to_list())

        # 토큰 검사
        prompt_tokens = count_tokens_fn(prompt)
        token_check = check_token_limit(
            prompt_tokens,
            max_context,
            params.max_tokens.value
        )

        match token_check:
            case Failure(err):
                return Failure(GenerationError(
                    reason=f"Token limit exceeded: {err.actual} > {err.limit}"
                ))
            case Success(_):
                pass

        # 생성 (I/O)
        result = generate_fn(prompt, params)

        # 응답 변환
        return (
            Railway.from_result(result)
            .map(lambda content: InferenceResponse(
                content=content,
                finish_reason="stop"
            ))
            .unwrap()
        )

    return workflow


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
    from mlx_llm_inference.streaming import mlx_stream_generator

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
            # Fallback
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
# 엔진 팩토리
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

        # 워크플로우 생성
        self._workflow = create_inference_workflow(
            generate_fn=self._generate_fn,
            count_tokens_fn=self._count_tokens_fn,
            apply_template_fn=self._apply_template_fn,
            max_context=max_context,
        )

    def generate(
        self,
        messages: NonEmptyList[Message],
        params: GenerationParams,
    ) -> Result[InferenceResponse, GenerationError]:
        """동기 생성"""
        return self._workflow(messages, params)

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

    @property
    def model_name(self) -> str:
        return self._bundle.name
```

## loader.py - 모델 로더

```python
"""모델 로더"""
from dataclasses import dataclass
from typing import Any
from functools import lru_cache

from mlx_llm_core import Result, Success, Failure, ModelNotFoundError


@dataclass(frozen=True)
class ModelBundle:
    """로드된 모델 번들"""
    name: str
    model: Any  # MLX model
    tokenizer: Any  # Tokenizer


def setup_custom_models() -> None:
    """커스텀 모델 등록"""
    import sys
    from mlx_llm_inference.custom_models import qwen3

    # MLX-LM이 찾을 수 있도록 등록
    sys.modules['mlx_lm.models.qwen3'] = qwen3


@lru_cache(maxsize=4)
def load_model(model_name: str) -> ModelBundle:
    """
    모델 로드 (캐시됨)

    최대 4개 모델까지 메모리에 유지합니다.
    """
    setup_custom_models()

    from mlx_lm import load

    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)
    print(f"Model loaded: {model_name}")

    return ModelBundle(
        name=model_name,
        model=model,
        tokenizer=tokenizer,
    )


def load_model_safe(model_name: str) -> Result[ModelBundle, ModelNotFoundError]:
    """모델 로드 (Result 반환)"""
    try:
        bundle = load_model(model_name)
        return Success(bundle)
    except Exception as e:
        return Failure(ModelNotFoundError(model=model_name))


def unload_model(model_name: str) -> None:
    """모델 언로드 (캐시에서 제거)"""
    load_model.cache_clear()


def list_loaded_models() -> list[str]:
    """로드된 모델 목록"""
    cache_info = load_model.cache_info()
    # Note: lru_cache doesn't expose keys, this is approximate
    return []  # 실제 구현에서는 별도 추적 필요
```

## streaming.py - 스트리밍 Generator

```python
"""스트리밍 Generator"""
from typing import Iterator, Any
import mlx.core as mx
from mlx_lm.utils import generate_step


def mlx_stream_generator(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterator[str]:
    """
    MLX 스트리밍 Generator

    UTF-8 멀티바이트 문자를 올바르게 처리합니다.
    토큰을 누적하고 완성된 문자만 yield합니다.
    """
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    token_buffer: list[int] = []
    prev_text = ""

    for token, _ in zip(
        generate_step(
            prompt_tokens,
            model,
            temp=temperature,
            top_p=top_p,
        ),
        range(max_tokens),
    ):
        # EOS 체크
        if token == tokenizer.eos_token_id:
            break

        # 토큰 ID 추출
        token_id = token[0] if isinstance(token, tuple) else token
        token_buffer.append(
            token_id.item() if hasattr(token_id, 'item') else token_id
        )

        # 누적 토큰 디코드
        try:
            current_text = tokenizer.decode(
                token_buffer,
                skip_special_tokens=True,
                errors='replace'
            )
        except Exception:
            continue

        # 새로운 텍스트만 추출
        new_text = current_text[len(prev_text):]

        # 불완전한 UTF-8 시퀀스 체크 (replacement character)
        if new_text and '\ufffd' not in new_text:
            yield new_text
            prev_text = current_text


def chunk_stream(
    stream: Iterator[str],
    min_chunk_size: int = 1,
) -> Iterator[str]:
    """
    스트림 청크 조절

    너무 작은 청크를 모아서 전송합니다.
    """
    buffer = ""

    for chunk in stream:
        buffer += chunk

        if len(buffer) >= min_chunk_size:
            yield buffer
            buffer = ""

    if buffer:
        yield buffer
```

## registry.py - 모델 레지스트리

```python
"""모델 레지스트리"""
from dataclasses import dataclass
from typing import Callable, Any
from functools import lru_cache


@dataclass(frozen=True)
class ModelInfo:
    """모델 정보"""
    name: str
    model_type: str
    quantization: str
    context_length: int


# 레지스트리 (딕셔너리)
_model_registry: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> None:
    """모델 등록"""
    _model_registry[info.name] = info


def get_model_info(name: str) -> ModelInfo | None:
    """모델 정보 조회"""
    return _model_registry.get(name)


def list_registered_models() -> list[ModelInfo]:
    """등록된 모델 목록"""
    return list(_model_registry.values())


# 기본 모델 등록
def _register_defaults():
    register_model(ModelInfo(
        name="mlx-community/Qwen3-8B-4bit",
        model_type="qwen3",
        quantization="4bit",
        context_length=8192,
    ))
    register_model(ModelInfo(
        name="mlx-community/Qwen2.5-7B-Instruct-4bit",
        model_type="qwen2",
        quantization="4bit",
        context_length=32768,
    ))

_register_defaults()
```

## Public API (__init__.py)

```python
"""MLX LLM Inference - MLX Integration Layer"""
from mlx_llm_inference.engine import (
    InferenceEngine,
    create_inference_workflow,
    # Function types
    GenerateFn, StreamFn, TokenCountFn, TemplateFn,
)
from mlx_llm_inference.loader import (
    ModelBundle,
    load_model, load_model_safe, unload_model,
)
from mlx_llm_inference.streaming import (
    mlx_stream_generator, chunk_stream,
)
from mlx_llm_inference.registry import (
    ModelInfo,
    register_model, get_model_info, list_registered_models,
)

__version__ = "0.1.0"

__all__ = [
    # Engine
    "InferenceEngine",
    "create_inference_workflow",
    # Types
    "GenerateFn", "StreamFn", "TokenCountFn", "TemplateFn",
    # Loader
    "ModelBundle",
    "load_model", "load_model_safe", "unload_model",
    # Streaming
    "mlx_stream_generator", "chunk_stream",
    # Registry
    "ModelInfo",
    "register_model", "get_model_info", "list_registered_models",
]
```

## 사용 예시

```python
from mlx_llm_core import (
    Message, GenerationParams, NonEmptyList,
    Temperature, TopP, MaxTokens,
)
from mlx_llm_inference import InferenceEngine

# 엔진 생성
engine = InferenceEngine("mlx-community/Qwen3-8B-4bit")

# 메시지 준비
messages = NonEmptyList.of([
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="Hello!"),
]).unwrap_or_raise()

# 파라미터
params = GenerationParams(
    max_tokens=MaxTokens(1024),
    temperature=Temperature(0.7),
    top_p=TopP(0.9),
)

# 동기 생성
result = engine.generate(messages, params)
match result:
    case Success(response):
        print(response.content)
    case Failure(error):
        print(f"Error: {error}")

# 스트리밍
for chunk in engine.stream(messages, params):
    print(chunk, end="", flush=True)
```
