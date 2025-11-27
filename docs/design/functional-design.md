# 함수형 디자인 원칙 - Python + MLX + Poetry

Scott Wlaschin의 함수형 디자인 원칙을 Python/MLX/Poetry 환경에 적용한 가이드입니다.

## 핵심 철학

### OOP vs FP 패턴 매핑

| OOP 패턴 | Python FP 대안 | 예시 |
|----------|----------------|------|
| Strategy | 일급 함수 | `process(data, strategy_fn)` |
| Decorator | 함수 합성 / `@decorator` | `@validate` |
| Factory | 순수 함수 | `create_model(config)` |
| Singleton | 모듈 레벨 상수 | `settings = Settings()` |
| Iterator | Generator | `yield token` |

### Python에서의 함수형 원칙

```python
# 1. 불변성 (Immutability)
from dataclasses import dataclass, frozen=True
from pydantic import BaseModel  # 기본적으로 불변

@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float = 0.7

# 2. 순수 함수 (Pure Functions)
def calculate_tokens(text: str, tokenizer) -> int:
    """부작용 없는 순수 함수"""
    return len(tokenizer.encode(text))

# 3. 함수 합성 (Composition)
from functools import reduce, partial

pipeline = [tokenize, normalize, encode]
result = reduce(lambda x, f: f(x), pipeline, input_text)
```

## MLX 프레임워크 적용

### Apple Silicon 최적화 함수형 패턴

```python
import mlx.core as mx
from typing import Iterator, Callable

# 지연 평가 (Lazy Evaluation) - MLX의 핵심
def create_inference_pipeline(
    model: mx.Module,
    tokenizer,
) -> Callable[[str], Iterator[str]]:
    """고차 함수로 추론 파이프라인 생성"""

    def infer(prompt: str) -> Iterator[str]:
        tokens = mx.array(tokenizer.encode(prompt))
        for token in generate_step(tokens, model):
            yield tokenizer.decode([token])

    return infer

# 사용
inference = create_inference_pipeline(model, tokenizer)
for chunk in inference("Hello"):
    print(chunk)
```

### Generator 기반 스트리밍

```python
def stream_tokens(
    prompt_tokens: mx.array,
    model: mx.Module,
    max_tokens: int,
) -> Iterator[int]:
    """
    Generator는 Python의 핵심 함수형 패턴.
    - 지연 평가
    - 메모리 효율
    - 합성 가능
    """
    for token, _ in zip(
        generate_step(prompt_tokens, model),
        range(max_tokens),
    ):
        if token == EOS_TOKEN:
            return
        yield token
```

## Poetry 프로젝트 구조

```toml
# pyproject.toml - 선언적 의존성 관리
[tool.poetry]
name = "mlx-llm-server"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.10,<3.13"  # MLX 제약
mlx = "^0.21.0"
mlx-lm = "^0.21.0"
pydantic = "^2.0"       # 타입 주도 설계

[tool.poetry.scripts]
mlx-llm-server = "mlx_llm_server.cli:main"  # 진입점
```

### 함수형 모듈 구조

```
src/mlx_llm_server/
├── config.py          # 불변 설정 (frozen dataclass)
├── models.py          # Pydantic 타입 정의
├── inference.py       # 순수 함수 + Generator
├── app.py             # 합성된 엔드포인트
└── custom_models/     # 확장 가능한 모델 등록
    └── qwen3.py
```

## 실제 적용 예시

### Before: 명령형 스타일

```python
# 나쁜 예: 상태 변경, 부작용 혼재
class InferenceEngine:
    def __init__(self):
        self.result = ""
        self.tokens = []

    def generate(self, prompt):
        self.tokens = self.tokenize(prompt)  # 상태 변경
        for token in self.model.generate(self.tokens):
            self.result += self.decode(token)  # 상태 변경
            self.log(token)  # 부작용
        return self.result
```

### After: 함수형 스타일

```python
# 좋은 예: 불변, 순수 함수, Generator
def generate_stream(
    prompt: str,
    model: mx.Module,
    tokenizer,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """순수 함수 + Generator로 스트리밍"""
    tokens = mx.array(tokenizer.encode(prompt))

    for token in generate_step(tokens, model):
        if token == tokenizer.eos_token_id:
            return
        yield tokenizer.decode([token])

# 합성하여 사용
def create_chat_response(
    messages: list[Message],
    generate_fn: Callable[[str], Iterator[str]],
) -> Iterator[ChatChunk]:
    prompt = apply_template(messages)
    for chunk in generate_fn(prompt):
        yield ChatChunk(content=chunk)
```

## 다음 문서

- [Railway Oriented Programming](./railway-oriented-programming.md) - 에러 처리 패턴
- [Type-Driven Design](./type-driven-design.md) - Pydantic 타입 주도 설계
- [Composition Patterns](./composition-patterns.md) - 함수 합성 패턴
