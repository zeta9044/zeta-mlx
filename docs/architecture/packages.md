# 패키지 구조

Poetry Workspace 기반 멀티 패키지 구조입니다.

## 디렉토리 레이아웃

```
mlx-llm/
├── pyproject.toml                    # Workspace 루트
├── README.md
├── CLAUDE.md
│
├── packages/
│   │
│   ├── mlx-llm-core/                 # ════════════════════════════
│   │   ├── pyproject.toml            # 순수 도메인 레이어
│   │   └── src/                      #
│   │       └── mlx_llm_core/         # 타입, Result, 순수 함수
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── types.py          # 도메인 타입 (Message, Params)
│   │           ├── result.py         # Result[T, E], Railway
│   │           ├── errors.py         # 에러 타입 (OR 타입)
│   │           ├── validation.py     # 순수 검증 함수
│   │           ├── pipeline.py       # pipe, compose 유틸
│   │           └── config.py         # 설정 타입 (Pydantic)
│   │
│   ├── mlx-llm-inference/            # ════════════════════════════
│   │   ├── pyproject.toml            # MLX 추론 레이어
│   │   └── src/                      #
│   │       └── mlx_llm_inference/    # 모델 로딩, 생성, 스트리밍
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── engine.py         # 추론 엔진 (generate 함수)
│   │           ├── loader.py         # 모델 로더 (load_model 함수)
│   │           ├── streaming.py      # 스트리밍 Generator
│   │           ├── tokenizer.py      # 토큰 카운터
│   │           └── custom_models/    # 커스텀 모델 (Qwen3 등)
│   │               ├── __init__.py
│   │               └── qwen3.py
│   │
│   ├── mlx-llm-api/                  # ════════════════════════════
│   │   ├── pyproject.toml            # HTTP API 레이어
│   │   └── src/                      #
│   │       └── mlx_llm_api/          # FastAPI, OpenAI 호환
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── app.py            # FastAPI 앱
│   │           ├── routes/           # 라우트 핸들러
│   │           │   ├── chat.py       # /v1/chat/completions
│   │           │   ├── models.py     # /v1/models
│   │           │   └── health.py     # /health
│   │           ├── dto/              # DTO (외부 세계)
│   │           │   ├── requests.py   # 요청 DTO
│   │           │   └── responses.py  # 응답 DTO
│   │           ├── converters.py     # DTO <-> Domain 변환
│   │           └── middleware.py     # CORS, 로깅
│   │
│   ├── mlx-llm-cli/                  # ════════════════════════════
│   │   ├── pyproject.toml            # CLI 레이어
│   │   └── src/                      #
│   │       └── mlx_llm_cli/          # Click 명령어
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── main.py           # CLI 진입점
│   │           ├── commands/         # 명령어 그룹
│   │           │   ├── serve.py      # mlx-llm serve
│   │           │   ├── chat.py       # mlx-llm chat
│   │           │   ├── rag.py        # mlx-llm rag
│   │           │   └── config.py     # mlx-llm config
│   │           └── formatters.py     # Rich 출력
│   │
│   ├── mlx-llm-rag/                  # ════════════════════════════
│   │   ├── pyproject.toml            # RAG 레이어
│   │   └── src/                      #
│   │       └── mlx_llm_rag/          # 문서 처리, 검색
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── chunker.py        # 문서 청킹
│   │           ├── embedder.py       # 임베딩 생성
│   │           ├── retriever.py      # 검색기
│   │           ├── pipeline.py       # RAG 파이프라인
│   │           └── vectorstore/      # 벡터 저장소
│   │               ├── __init__.py
│   │               ├── base.py       # Protocol 정의
│   │               └── faiss.py      # FAISS 구현
│   │
│   └── mlx-llm-langchain/            # ════════════════════════════
│       ├── pyproject.toml            # LangChain 어댑터
│       └── src/                      #
│           └── mlx_llm_langchain/    # LangChain 통합
│               ├── __init__.py       # ════════════════════════════
│               ├── chat_model.py     # BaseChatModel 구현
│               ├── embeddings.py     # Embeddings 구현
│               └── tools.py          # Tool 정의
│
├── tests/                            # 테스트
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
└── docs/
    ├── design/                       # 함수형 디자인 원칙
    └── architecture/                 # 아키텍처 문서
```

## pyproject.toml 구조

### 루트 (Workspace)

```toml
# /pyproject.toml
[tool.poetry]
name = "mlx-llm-workspace"
version = "0.1.0"
description = "MLX LLM Platform Workspace"
authors = ["ZetaLab <zeta@example.com>"]
packages = []  # 워크스페이스는 패키지 없음

[tool.poetry.dependencies]
python = "^3.10,<3.13"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.24"
mypy = "^1.13"
ruff = "^0.8"

# 워크스페이스 멤버 정의 (Poetry 1.2+)
[tool.poetry.workspace]
members = [
    "packages/mlx-llm-core",
    "packages/mlx-llm-inference",
    "packages/mlx-llm-api",
    "packages/mlx-llm-cli",
    "packages/mlx-llm-rag",
    "packages/mlx-llm-langchain",
]
```

### Core 패키지

```toml
# /packages/mlx-llm-core/pyproject.toml
[tool.poetry]
name = "mlx-llm-core"
version = "0.1.0"
description = "Core types and pure functions for MLX LLM"
packages = [{include = "mlx_llm_core", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
pydantic = "^2.9"
pydantic-settings = "^2.6"
pyyaml = "^6.0"
```

### Inference 패키지

```toml
# /packages/mlx-llm-inference/pyproject.toml
[tool.poetry]
name = "mlx-llm-inference"
version = "0.1.0"
description = "MLX inference engine"
packages = [{include = "mlx_llm_inference", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
mlx-llm-core = {path = "../mlx-llm-core", develop = true}
mlx = "^0.21"
mlx-lm = "^0.21"
```

### API 패키지

```toml
# /packages/mlx-llm-api/pyproject.toml
[tool.poetry]
name = "mlx-llm-api"
version = "0.1.0"
description = "FastAPI server for MLX LLM"
packages = [{include = "mlx_llm_api", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
mlx-llm-core = {path = "../mlx-llm-core", develop = true}
mlx-llm-inference = {path = "../mlx-llm-inference", develop = true}
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.32"}
```

### CLI 패키지

```toml
# /packages/mlx-llm-cli/pyproject.toml
[tool.poetry]
name = "mlx-llm-cli"
version = "0.1.0"
description = "CLI for MLX LLM"
packages = [{include = "mlx_llm_cli", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
mlx-llm-core = {path = "../mlx-llm-core", develop = true}
mlx-llm-inference = {path = "../mlx-llm-inference", develop = true}
mlx-llm-api = {path = "../mlx-llm-api", develop = true}
click = "^8.1"
rich = "^13.9"

[tool.poetry.scripts]
mlx-llm = "mlx_llm_cli.main:cli"
```

### RAG 패키지

```toml
# /packages/mlx-llm-rag/pyproject.toml
[tool.poetry]
name = "mlx-llm-rag"
version = "0.1.0"
description = "RAG pipeline for MLX LLM"
packages = [{include = "mlx_llm_rag", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
mlx-llm-core = {path = "../mlx-llm-core", develop = true}
faiss-cpu = "^1.8"
sentence-transformers = "^3.3"
```

### LangChain 패키지

```toml
# /packages/mlx-llm-langchain/pyproject.toml
[tool.poetry]
name = "mlx-llm-langchain"
version = "0.1.0"
description = "LangChain integration for MLX LLM"
packages = [{include = "mlx_llm_langchain", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
mlx-llm-core = {path = "../mlx-llm-core", develop = true}
mlx-llm-inference = {path = "../mlx-llm-inference", develop = true}
langchain-core = "^0.3"
```

## 의존성 규칙

### 허용된 의존성 방향

```
cli ──► api ──► inference ──► core
         │           │
         │           ▼
         └────► rag ──► core

langchain ──► inference ──► core
```

### 금지된 의존성

```
core ──✗──► inference    # Core는 외부 의존 금지
core ──✗──► api          # Core는 순수해야 함
inference ──✗──► api     # 역방향 금지
rag ──✗──► api           # 역방향 금지
```

## 설치 및 개발

```bash
# 전체 워크스페이스 설치
cd mlx-llm
poetry install

# 특정 패키지만 설치
cd packages/mlx-llm-core
poetry install

# 개발 모드로 전체 설치
poetry install --all-extras

# CLI 실행
poetry run mlx-llm serve
poetry run mlx-llm chat "Hello"
```

## 패키지별 익스포트

각 패키지는 `__init__.py`에서 public API만 노출합니다:

```python
# mlx_llm_core/__init__.py
from mlx_llm_core.types import Message, GenerationParams, Role
from mlx_llm_core.result import Result, Success, Failure, Railway
from mlx_llm_core.errors import ValidationError, InferenceError
from mlx_llm_core.validation import validate_messages, validate_params
from mlx_llm_core.pipeline import pipe, compose

__all__ = [
    # Types
    "Message", "GenerationParams", "Role",
    # Result
    "Result", "Success", "Failure", "Railway",
    # Errors
    "ValidationError", "InferenceError",
    # Functions
    "validate_messages", "validate_params",
    "pipe", "compose",
]
```
