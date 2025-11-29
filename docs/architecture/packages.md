# 패키지 구조

Poetry Workspace 기반 멀티 패키지 구조입니다.

## 디렉토리 레이아웃

```
zeta-mlx/
├── pyproject.toml                    # Workspace 루트
├── README.md
├── CLAUDE.md
│
├── packages/
│   │
│   ├── core/                         # ════════════════════════════
│   │   ├── pyproject.toml            # 순수 도메인 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx/             # 네임스페이스 패키지
│   │           └── core/             # 타입, Result, 순수 함수
│   │               ├── __init__.py   # ════════════════════════════
│   │               ├── types.py      # 도메인 타입 (Message, Params)
│   │               ├── result.py     # Result[T, E], Railway
│   │               ├── errors.py     # 에러 타입 (OR 타입)
│   │               ├── validation.py # 순수 검증 함수
│   │               ├── pipeline.py   # pipe, compose 유틸
│   │               └── config.py     # 설정 타입 (Pydantic)
│   │
│   ├── inference/                    # ════════════════════════════
│   │   ├── pyproject.toml            # MLX 추론 + API 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx/             # 네임스페이스 패키지
│   │           └── inference/        # 모델 로딩, 생성, API
│   │               ├── __init__.py   # ════════════════════════════
│   │               ├── engine.py     # 추론 엔진 (generate 함수)
│   │               ├── loader.py     # 모델 로더 (load_model 함수)
│   │               ├── streaming.py  # 스트리밍 Generator
│   │               ├── manager.py    # 모델 관리자
│   │               ├── custom_models/# 커스텀 모델 (Qwen3 등)
│   │               │   ├── __init__.py
│   │               │   └── qwen3.py
│   │               └── api/          # OpenAI/vLLM 호환 API
│   │                   ├── __init__.py
│   │                   ├── app.py    # FastAPI 앱
│   │                   ├── converters.py # DTO <-> Domain 변환
│   │                   ├── dto/      # DTO
│   │                   │   ├── requests.py
│   │                   │   └── responses.py
│   │                   └── routes/   # 라우트
│   │                       ├── chat.py     # /v1/chat/completions
│   │                       ├── models.py   # /v1/models
│   │                       ├── tokenize.py # /tokenize, /detokenize (vLLM)
│   │                       └── health.py   # /health
│   │
│   ├── embedding/                    # ════════════════════════════
│   │   ├── pyproject.toml            # 임베딩 + API 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx/             # 네임스페이스 패키지
│   │           └── embedding/        # 임베딩 모델, API
│   │               ├── __init__.py   # ════════════════════════════
│   │               ├── types.py      # 임베딩 타입
│   │               ├── errors.py     # 에러 타입
│   │               ├── engine.py     # 임베딩 엔진
│   │               ├── loader.py     # 모델 로더
│   │               └── api/          # OpenAI 호환 API
│   │                   ├── __init__.py
│   │                   ├── app.py    # FastAPI 앱
│   │                   ├── dto/      # DTO
│   │                   └── routes/   # 라우트
│   │
│   ├── cli/                          # ════════════════════════════
│   │   ├── pyproject.toml            # CLI 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx/             # 네임스페이스 패키지
│   │           └── cli/              # Typer 명령어
│   │               ├── __init__.py   # ════════════════════════════
│   │               ├── main.py       # CLI 진입점
│   │               ├── commands/     # 명령어 그룹
│   │               │   ├── serve.py  # zeta-mlx serve
│   │               │   ├── embedding.py # zeta-mlx embedding
│   │               │   ├── chat.py   # zeta-mlx chat
│   │               │   └── models.py # zeta-mlx models
│   │               └── formatters.py # Rich 출력
│   │
│   ├── rag/                          # ════════════════════════════
│   │   ├── pyproject.toml            # RAG 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx/             # 네임스페이스 패키지
│   │           └── rag/              # 문서 처리, 검색
│   │               ├── __init__.py   # ════════════════════════════
│   │               ├── types.py      # RAG 타입
│   │               ├── embeddings.py # 임베딩 생성
│   │               ├── retriever.py  # 검색기
│   │               ├── vector_store.py # 벡터 저장소
│   │               └── rag_pipeline.py # RAG 파이프라인
│   │
│   └── langchain/                    # ════════════════════════════
│       ├── pyproject.toml            # LangChain 어댑터
│       └── src/                      #
│           └── zeta_mlx/             # 네임스페이스 패키지
│               └── langchain/        # LangChain 통합
│                   ├── __init__.py   # ════════════════════════════
│                   ├── chat_model.py # BaseChatModel 구현
│                   ├── embeddings.py # Embeddings 구현
│                   └── tools.py      # Tool 정의
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

## Import 방식

네임스페이스 패키지 패턴을 사용합니다:

```python
# Core
from zeta_mlx.core import Message, GenerationParams, Result, Success, Failure

# Inference
from zeta_mlx.inference import InferenceEngine, load_model_safe

# Embedding
from zeta_mlx.embedding import EmbeddingEngine, create_embedding_engine

# CLI
from zeta_mlx.cli import cli

# RAG
from zeta_mlx.rag import RAGPipeline, create_retriever

# LangChain
from zeta_mlx.langchain import ChatMLXLLM
```

## pyproject.toml 구조

### 루트 (Workspace)

```toml
# /pyproject.toml
[tool.poetry]
name = "zeta-mlx"
version = "0.1.0"
description = "Zeta MLX Platform"
packages = []  # 워크스페이스는 패키지 없음

[tool.poetry.dependencies]
python = "^3.10,<3.13"

# Workspace packages
zeta-mlx-core = {path = "packages/core", develop = true}
zeta-mlx-inference = {path = "packages/inference", develop = true}
zeta-mlx-cli = {path = "packages/cli", develop = true}
zeta-mlx-embedding = {path = "packages/embedding", develop = true, optional = true}
zeta-mlx-rag = {path = "packages/rag", develop = true, optional = true}
zeta-mlx-langchain = {path = "packages/langchain", develop = true, optional = true}

[tool.poetry.extras]
embedding = ["zeta-mlx-embedding"]
rag = ["zeta-mlx-rag"]
langchain = ["zeta-mlx-langchain"]
all = ["zeta-mlx-embedding", "zeta-mlx-rag", "zeta-mlx-langchain"]

[tool.poetry.scripts]
zeta-mlx = "zeta_mlx.cli:cli"
```

### Core 패키지

```toml
# /packages/core/pyproject.toml
[tool.poetry]
name = "zeta-mlx-core"
version = "0.1.0"
description = "Core types and pure functions for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
pydantic = "^2.9"
pydantic-settings = "^2.6"
pyyaml = "^6.0"
```

### Inference 패키지

```toml
# /packages/inference/pyproject.toml
[tool.poetry]
name = "zeta-mlx-inference"
version = "0.1.0"
description = "MLX inference engine and API for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../core", develop = true}
mlx = "^0.30"
mlx-lm = "^0.28"
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.32"}
```

### Embedding 패키지

```toml
# /packages/embedding/pyproject.toml
[tool.poetry]
name = "zeta-mlx-embedding"
version = "0.1.0"
description = "Embedding model serving for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../core", develop = true}
numpy = "^1.26"
sentence-transformers = {version = "^3.0", optional = true}
fastapi = "^0.115"
uvicorn = "^0.32"

[tool.poetry.extras]
sentence-transformers = ["sentence-transformers"]
```

### CLI 패키지

```toml
# /packages/cli/pyproject.toml
[tool.poetry]
name = "zeta-mlx-cli"
version = "0.1.0"
description = "CLI for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../core", develop = true}
zeta-mlx-inference = {path = "../inference", develop = true}
zeta-mlx-embedding = {path = "../embedding", develop = true}
typer = {extras = ["all"], version = "^0.16"}
rich = "^13.7"

[tool.poetry.scripts]
zeta-mlx = "zeta_mlx.cli:cli"
```

### RAG 패키지

```toml
# /packages/rag/pyproject.toml
[tool.poetry]
name = "zeta-mlx-rag"
version = "0.1.0"
description = "RAG pipeline for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../core", develop = true}
zeta-mlx-inference = {path = "../inference", develop = true}
numpy = "^1.26"
sentence-transformers = {version = "^3.0", optional = true}
```

### LangChain 패키지

```toml
# /packages/langchain/pyproject.toml
[tool.poetry]
name = "zeta-mlx-langchain"
version = "0.1.0"
description = "LangChain integration for Zeta MLX"
packages = [{include = "zeta_mlx", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../core", develop = true}
zeta-mlx-inference = {path = "../inference", develop = true}
langchain-core = "^1.1"
```

## 의존성 규칙

### 허용된 의존성 방향

```
cli ──► inference ──► core
             │
             ▼
langchain ──► inference ──► core

embedding ──► core

rag ──► core
```

### 금지된 의존성

```
core ──✗──► inference    # Core는 외부 의존 금지
core ──✗──► embedding    # Core는 순수해야 함
inference ──✗──► cli     # 역방향 금지
```

## 설치 및 개발

```bash
# 전체 워크스페이스 설치
cd zeta-mlx
poetry install

# 특정 패키지만 설치
cd packages/core
poetry install

# 개발 모드로 전체 설치
poetry install --all-extras

# CLI 실행
poetry run zeta-mlx serve
poetry run zeta-mlx chat "Hello"
```

## 패키지별 익스포트

각 패키지는 `__init__.py`에서 public API만 노출합니다:

```python
# zeta_mlx/core/__init__.py
from zeta_mlx.core.types import Message, GenerationParams, Role
from zeta_mlx.core.result import Result, Success, Failure, Railway
from zeta_mlx.core.errors import ValidationError, InferenceError
from zeta_mlx.core.validation import validate_messages, validate_params
from zeta_mlx.core.pipeline import pipe, compose

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
