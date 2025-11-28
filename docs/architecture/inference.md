# Inference 패키지 (zeta-mlx-inference)

MLX 기반 추론 엔진입니다. I/O 경계에 위치하며, Core의 순수 타입을 사용합니다.

## 설계 원칙

- **I/O at the Edge**: MLX 호출은 이 레이어에서만
- **Return Result**: 모든 함수는 `Result[T, E]` 반환
- **Dependency Injection**: 의존성은 함수 파라미터로 주입
- **Generator for Streaming**: 스트리밍은 Generator 패턴

## 모듈 구조

```
zeta_mlx_inference/
├── __init__.py       # Public API
├── engine.py         # 추론 엔진 (단일 모델)
├── manager.py        # 다중 모델 관리자 (LRU)
├── loader.py         # 모델 로더
├── streaming.py      # 스트리밍 Generator
├── tokenizer.py      # 토큰 카운터
└── custom_models/    # 커스텀 모델
    ├── __init__.py
    └── qwen3.py
```

## engine.py - 추론 엔진

```python
"""MLX 추론 엔진"""
from typing import Iterator, Callable
from functools import partial

from zeta_mlx_core import (
    Result, Success, Failure, Railway,
    Message, GenerationParams, InferenceResponse,
    GenerationError, TokenLimitError,
    validate_messages, check_token_limit,
    NonEmptyList,
)
from zeta_mlx_inference.loader import load_model, ModelBundle
from zeta_mlx_inference.streaming import create_stream_generator
from zeta_mlx_inference.tokenizer import count_tokens, apply_chat_template


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
    from zeta_mlx_inference.streaming import mlx_stream_generator

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

from zeta_mlx_core import Result, Success, Failure, ModelNotFoundError


@dataclass(frozen=True)
class ModelBundle:
    """로드된 모델 번들"""
    name: str
    model: Any  # MLX model
    tokenizer: Any  # Tokenizer


def setup_custom_models() -> None:
    """커스텀 모델 등록"""
    import sys
    from zeta_mlx_inference.custom_models import qwen3

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

## manager.py - 다중 모델 관리자

```python
"""다중 모델 관리자 (LRU 기반)"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator
from threading import Lock

from zeta_mlx_core import (
    Result, Success, Failure,
    ModelsConfig, ModelDefinition,
    ModelNotFoundError, GenerationError,
    Message, GenerationParams, InferenceResponse,
)
from zeta_mlx_inference.loader import ModelBundle, load_model_safe
from zeta_mlx_inference.engine import InferenceEngine


@dataclass
class LoadedModel:
    """로드된 모델 정보"""
    alias: str
    definition: ModelDefinition
    engine: InferenceEngine


class ModelManager:
    """
    다중 모델 관리자

    LRU 방식으로 모델을 관리하며, 최대 로드 수를 초과하면
    가장 오래 사용되지 않은 모델을 언로드합니다.
    """

    def __init__(self, config: ModelsConfig):
        self._config = config
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = Lock()

    @property
    def default_alias(self) -> str:
        """기본 모델 별칭"""
        return self._config.default

    def list_available(self) -> list[str]:
        """사용 가능한 모델 별칭 목록"""
        return self._config.list_aliases()

    def list_loaded(self) -> list[str]:
        """현재 로드된 모델 별칭 목록"""
        with self._lock:
            return list(self._loaded.keys())

    def get_model_info(self, alias: str) -> ModelDefinition | None:
        """모델 정의 조회"""
        return self._config.get_model(alias)

    def resolve_alias(self, model_name: str) -> str:
        """
        모델 이름을 별칭으로 해석

        - 빈 문자열 또는 None → 기본 모델
        - 별칭이 존재하면 그대로 사용
        - HuggingFace 경로면 역방향 조회
        """
        if not model_name:
            return self._config.default

        # 별칭으로 직접 존재하는지
        if model_name in self._config.available:
            return model_name

        # HuggingFace 경로로 역방향 조회
        for alias, defn in self._config.available.items():
            if defn.path == model_name:
                return alias

        # 찾지 못하면 그대로 반환 (에러는 호출자가 처리)
        return model_name

    def get_engine(self, alias: str) -> Result[InferenceEngine, ModelNotFoundError]:
        """
        모델 엔진 가져오기 (필요시 로드)

        LRU 방식으로 최근 사용된 모델을 유지합니다.
        """
        resolved = self.resolve_alias(alias)

        with self._lock:
            # 이미 로드된 경우: LRU 갱신
            if resolved in self._loaded:
                self._loaded.move_to_end(resolved)
                return Success(self._loaded[resolved].engine)

            # 모델 정의 확인
            definition = self._config.get_model(resolved)
            if definition is None:
                return Failure(ModelNotFoundError(model=resolved))

            # 최대 로드 수 초과 시 가장 오래된 모델 언로드
            while len(self._loaded) >= self._config.max_loaded:
                oldest_alias, oldest = self._loaded.popitem(last=False)
                print(f"Unloading model: {oldest_alias}")
                del oldest  # GC에서 메모리 해제

            # 모델 로드
            print(f"Loading model: {resolved} ({definition.path})")
            bundle_result = load_model_safe(definition.path)

            if isinstance(bundle_result, Failure):
                return Failure(ModelNotFoundError(model=definition.path))

            engine = InferenceEngine(
                bundle_result.value,
                max_context=definition.context_length,
            )

            self._loaded[resolved] = LoadedModel(
                alias=resolved,
                definition=definition,
                engine=engine,
            )

            return Success(engine)

    def generate(
        self,
        alias: str,
        messages: list[Message],
        params: GenerationParams,
    ) -> Result[InferenceResponse, GenerationError | ModelNotFoundError]:
        """지정된 모델로 생성"""
        engine_result = self.get_engine(alias)

        if isinstance(engine_result, Failure):
            return engine_result

        return engine_result.value.generate(messages, params)

    def stream(
        self,
        alias: str,
        messages: list[Message],
        params: GenerationParams,
    ) -> Iterator[str]:
        """지정된 모델로 스트리밍 생성"""
        engine_result = self.get_engine(alias)

        if isinstance(engine_result, Failure):
            yield f"[Error: Model '{alias}' not found]"
            return

        yield from engine_result.value.stream(messages, params)

    def preload(self, aliases: list[str]) -> dict[str, bool]:
        """지정된 모델들 미리 로드"""
        results = {}
        for alias in aliases:
            result = self.get_engine(alias)
            results[alias] = isinstance(result, Success)
        return results

    def unload(self, alias: str) -> bool:
        """모델 명시적 언로드"""
        with self._lock:
            if alias in self._loaded:
                del self._loaded[alias]
                return True
            return False

    def unload_all(self) -> None:
        """모든 모델 언로드"""
        with self._lock:
            self._loaded.clear()


# ============================================================
# 팩토리 함수
# ============================================================

def create_model_manager(config: ModelsConfig) -> ModelManager:
    """모델 관리자 생성"""
    return ModelManager(config)


def create_model_manager_from_yaml(config_path: str) -> Result[ModelManager, str]:
    """YAML 설정에서 모델 관리자 생성"""
    from zeta_mlx_core import load_config

    config_result = load_config(config_path)
    if isinstance(config_result, Failure):
        return Failure(str(config_result.error))

    return Success(ModelManager(config_result.value.models))
```

## 다중 모델 사용 예시

```python
from zeta_mlx_core import load_config, Message, GenerationParams
from zeta_mlx_inference import create_model_manager

# 설정 로드
config = load_config("config.yaml").unwrap_or_raise()

# 모델 관리자 생성
manager = create_model_manager(config.models)

# 사용 가능한 모델 확인
print(manager.list_available())  # ['qwen3-8b', 'qwen3-4b', 'qwen2.5-7b', ...]

# 모델 미리 로드 (선택적)
manager.preload(["qwen3-8b"])

# 기본 모델로 생성
messages = [Message(role="user", content="Hello!")]
params = GenerationParams.default()

result = manager.generate("", messages, params)  # 빈 문자열 = 기본 모델

# 특정 모델로 생성
result = manager.generate("qwen2.5-7b", messages, params)

# 스트리밍
for chunk in manager.stream("qwen3-4b", messages, params):
    print(chunk, end="", flush=True)

# 현재 로드된 모델 확인
print(manager.list_loaded())  # ['qwen3-8b', 'qwen2.5-7b', ...]
```

## Public API (__init__.py)

```python
"""Zeta MLX Inference - MLX Integration Layer"""
from zeta_mlx_inference.engine import (
    InferenceEngine,
    create_inference_workflow,
    # Function types
    GenerateFn, StreamFn, TokenCountFn, TemplateFn,
)
from zeta_mlx_inference.manager import (
    ModelManager, LoadedModel,
    create_model_manager, create_model_manager_from_yaml,
)
from zeta_mlx_inference.loader import (
    ModelBundle,
    load_model, load_model_safe, unload_model,
)
from zeta_mlx_inference.streaming import (
    mlx_stream_generator, chunk_stream,
)

__version__ = "0.1.0"

__all__ = [
    # Engine (단일 모델)
    "InferenceEngine",
    "create_inference_workflow",
    # Manager (다중 모델)
    "ModelManager",
    "LoadedModel",
    "create_model_manager",
    "create_model_manager_from_yaml",
    # Types
    "GenerateFn", "StreamFn", "TokenCountFn", "TemplateFn",
    # Loader
    "ModelBundle",
    "load_model", "load_model_safe", "unload_model",
    # Streaming
    "mlx_stream_generator", "chunk_stream",
]
```

## 사용 예시

```python
from zeta_mlx_core import (
    Message, GenerationParams, NonEmptyList,
    Temperature, TopP, MaxTokens,
)
from zeta_mlx_inference import InferenceEngine

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
