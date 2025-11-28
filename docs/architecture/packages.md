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
│   ├── zeta-mlx-core/                 # ════════════════════════════
│   │   ├── pyproject.toml            # 순수 도메인 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_core/         # 타입, Result, 순수 함수
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── types.py          # 도메인 타입 (Message, Params)
│   │           ├── result.py         # Result[T, E], Railway
│   │           ├── errors.py         # 에러 타입 (OR 타입)
│   │           ├── validation.py     # 순수 검증 함수
│   │           ├── pipeline.py       # pipe, compose 유틸
│   │           └── config.py         # 설정 타입 (Pydantic)
│   │
│   ├── zeta-mlx-inference/            # ════════════════════════════
│   │   ├── pyproject.toml            # MLX 추론 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_inference/    # 모델 로딩, 생성, 스트리밍
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── engine.py         # 추론 엔진 (generate 함수)
│   │           ├── loader.py         # 모델 로더 (load_model 함수)
│   │           ├── streaming.py      # 스트리밍 Generator
│   │           ├── tokenizer.py      # 토큰 카운터
│   │           └── custom_models/    # 커스텀 모델 (Qwen3 등)
│   │               ├── __init__.py
│   │               └── qwen3.py
│   │
│   ├── zeta-mlx-embedding/            # ════════════════════════════
│   │   ├── pyproject.toml            # 임베딩 서빙 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_embedding/    # 임베딩 모델, API
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── types.py          # 임베딩 타입
│   │           ├── errors.py         # 에러 타입
│   │           ├── engine.py         # 임베딩 엔진
│   │           ├── loader.py         # 모델 로더
│   │           └── api/              # OpenAI 호환 API
│   │               ├── __init__.py
│   │               ├── app.py        # FastAPI 앱
│   │               ├── dto/          # DTO
│   │               └── routes/       # 라우트
│   │
│   ├── zeta-mlx-api/                  # ════════════════════════════
│   │   ├── pyproject.toml            # HTTP API 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_api/          # FastAPI, OpenAI 호환
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
│   ├── zeta-mlx-cli/                  # ════════════════════════════
│   │   ├── pyproject.toml            # CLI 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_cli/          # Click 명령어
│   │           ├── __init__.py       # ════════════════════════════
│   │           ├── main.py           # CLI 진입점
│   │           ├── commands/         # 명령어 그룹
│   │           │   ├── serve.py      # zeta-mlx serve
│   │           │   ├── chat.py       # zeta-mlx chat
│   │           │   ├── rag.py        # zeta-mlx rag
│   │           │   └── config.py     # zeta-mlx config
│   │           └── formatters.py     # Rich 출력
│   │
│   ├── zeta-mlx-rag/                  # ════════════════════════════
│   │   ├── pyproject.toml            # RAG 레이어
│   │   └── src/                      #
│   │       └── zeta_mlx_rag/          # 문서 처리, 검색
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
│   └── zeta-mlx-langchain/            # ════════════════════════════
│       ├── pyproject.toml            # LangChain 어댑터
│       └── src/                      #
│           └── zeta_mlx_langchain/    # LangChain 통합
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
name = "zeta-mlx-workspace"
version = "0.1.0"
description = "Zeta MLX Platform Workspace"
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
    "packages/zeta-mlx-core",
    "packages/zeta-mlx-inference",
    "packages/zeta-mlx-embedding",
    "packages/zeta-mlx-api",
    "packages/zeta-mlx-cli",
    "packages/zeta-mlx-rag",
    "packages/zeta-mlx-langchain",
]
```

### Core 패키지

```toml
# /packages/zeta-mlx-core/pyproject.toml
[tool.poetry]
name = "zeta-mlx-core"
version = "0.1.0"
description = "Core types and pure functions for Zeta MLX"
packages = [{include = "zeta_mlx_core", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
pydantic = "^2.9"
pydantic-settings = "^2.6"
pyyaml = "^6.0"
```

### Inference 패키지

```toml
# /packages/zeta-mlx-inference/pyproject.toml
[tool.poetry]
name = "zeta-mlx-inference"
version = "0.1.0"
description = "MLX inference engine"
packages = [{include = "zeta_mlx_inference", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
mlx = "^0.21"
mlx-lm = "^0.21"
```

### Embedding 패키지

```toml
# /packages/zeta-mlx-embedding/pyproject.toml
[tool.poetry]
name = "zeta-mlx-embedding"
version = "0.1.0"
description = "Embedding model serving for Zeta MLX"
packages = [{include = "zeta_mlx_embedding", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
numpy = "^1.26"
sentence-transformers = {version = "^3.0", optional = true}
fastapi = "^0.115"
uvicorn = "^0.32"

[tool.poetry.extras]
sentence-transformers = ["sentence-transformers"]
```

### API 패키지

```toml
# /packages/zeta-mlx-api/pyproject.toml
[tool.poetry]
name = "zeta-mlx-api"
version = "0.1.0"
description = "FastAPI server for Zeta MLX"
packages = [{include = "zeta_mlx_api", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
zeta-mlx-inference = {path = "../zeta-mlx-inference", develop = true}
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.32"}
```

### CLI 패키지

```toml
# /packages/zeta-mlx-cli/pyproject.toml
[tool.poetry]
name = "zeta-mlx-cli"
version = "0.1.0"
description = "CLI for Zeta MLX"
packages = [{include = "zeta_mlx_cli", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
zeta-mlx-inference = {path = "../zeta-mlx-inference", develop = true}
zeta-mlx-api = {path = "../zeta-mlx-api", develop = true}
click = "^8.1"
rich = "^13.9"

[tool.poetry.scripts]
zeta-mlx = "zeta_mlx_cli.main:cli"
```

### RAG 패키지

```toml
# /packages/zeta-mlx-rag/pyproject.toml
[tool.poetry]
name = "zeta-mlx-rag"
version = "0.1.0"
description = "RAG pipeline for Zeta MLX"
packages = [{include = "zeta_mlx_rag", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
faiss-cpu = "^1.8"
sentence-transformers = "^3.3"
```

### LangChain 패키지

```toml
# /packages/zeta-mlx-langchain/pyproject.toml
[tool.poetry]
name = "zeta-mlx-langchain"
version = "0.1.0"
description = "LangChain integration for Zeta MLX"
packages = [{include = "zeta_mlx_langchain", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10,<3.13"
zeta-mlx-core = {path = "../zeta-mlx-core", develop = true}
zeta-mlx-inference = {path = "../zeta-mlx-inference", develop = true}
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
cd zeta-mlx
poetry install

# 특정 패키지만 설치
cd packages/zeta-mlx-core
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
# zeta_mlx_core/__init__.py
from zeta_mlx_core.types import Message, GenerationParams, Role
from zeta_mlx_core.result import Result, Success, Failure, Railway
from zeta_mlx_core.errors import ValidationError, InferenceError
from zeta_mlx_core.validation import validate_messages, validate_params
from zeta_mlx_core.pipeline import pipe, compose

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
