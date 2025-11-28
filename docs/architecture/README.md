# MLX LLM Platform Architecture

Apple Silicon MLX 기반 LLM 서빙 플랫폼의 함수형 아키텍처입니다.

## 설계 원칙

이 아키텍처는 [Scott Wlaschin의 함수형 디자인 원칙](../design/README.md)을 따릅니다:

1. **Functions are Things** - 패키지는 함수를 노출, 클래스 최소화
2. **Composition Everywhere** - 패키지 간 파이프라인 합성
3. **Make Illegal States Unrepresentable** - 타입으로 제약 표현
4. **Railway Oriented Programming** - `Result[T, E]`로 에러 처리
5. **Onion Architecture** - Pure Core, Impure Edge

## 패키지 구조

```
zeta-mlx/
├── pyproject.toml                    # Workspace 루트
├── packages/
│   ├── zeta-mlx-core/                 # 순수 도메인 (Pure)
│   ├── zeta-mlx-inference/            # MLX 추론 (Impure)
│   ├── zeta-mlx-api/                  # HTTP API (Impure)
│   ├── zeta-mlx-cli/                  # CLI (Impure)
│   ├── zeta-mlx-rag/                  # RAG 파이프라인 (Impure)
│   └── zeta-mlx-langchain/            # LangChain 어댑터 (Impure)
└── docs/
    ├── design/                       # 함수형 디자인 원칙
    └── architecture/                 # 아키텍처 문서
```

## 패키지 의존성 (Onion 구조)

```
                    ┌─────────────────────────────────────┐
                    │           I/O Edge (Impure)         │
                    │  ┌─────┐  ┌─────┐  ┌───────────┐   │
                    │  │ cli │  │ api │  │ langchain │   │
                    │  └──┬──┘  └──┬──┘  └─────┬─────┘   │
                    │     │       │           │          │
                    │     └───────┼───────────┘          │
                    │             │                      │
                    │     ┌───────┴───────┐              │
                    │     ▼               ▼              │
                    │  ┌─────────┐  ┌─────────┐         │
                    │  │inference│  │   rag   │         │
                    │  └────┬────┘  └────┬────┘         │
                    │       │            │              │
                    │       └──────┬─────┘              │
                    │              │                    │
                    └──────────────┼────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │         Domain Core (Pure)        │
                    │         ┌──────────┐              │
                    │         │   core   │              │
                    │         │          │              │
                    │         │ - Types  │              │
                    │         │ - Result │              │
                    │         │ - Pipes  │              │
                    │         └──────────┘              │
                    └───────────────────────────────────┘
```

## 패키지별 책임

| 패키지 | 레이어 | 책임 | 의존성 |
|--------|--------|------|--------|
| `zeta-mlx-core` | Domain (Pure) | 타입, Result, 순수 함수 | 없음 |
| `zeta-mlx-inference` | Application | MLX 모델 로딩, 추론 | core |
| `zeta-mlx-rag` | Application | 문서 처리, 임베딩, 검색 | core |
| `zeta-mlx-api` | I/O Edge | FastAPI, OpenAI 호환 API | core, inference |
| `zeta-mlx-cli` | I/O Edge | Click CLI | core, inference, api |
| `zeta-mlx-langchain` | I/O Edge | LangChain 어댑터 | core, inference |

## 데이터 흐름

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ API Layer (Impure)                                          │
│                                                             │
│  ChatRequestDTO ──► to_domain() ──► ValidatedRequest        │
│       (JSON)         (검증)           (Domain Type)         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Workflow Pipeline (Pure)                                    │
│                                                             │
│  Railway.of(request)                                        │
│    .bind(validate_messages)     # Result[_, ValidationError]│
│    .bind(check_token_limit)     # Result[_, TokenLimitError]│
│    .map(format_prompt)          # 순수 변환                  │
│    .unwrap()                                                │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ Inference Layer (Impure)                                    │
│                                                             │
│  generate(prompt) ──► MLX Model ──► Response                │
│                         (I/O)                               │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ API Layer (Impure)                                          │
│                                                             │
│  InferenceResponse ──► to_dto() ──► ChatResponseDTO         │
│    (Domain Type)                       (JSON)               │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
HTTP Response
```

## 문서 목록

| 문서 | 설명 |
|------|------|
| [packages.md](./packages.md) | 패키지별 상세 구조 |
| [core.md](./core.md) | Core 패키지 설계 |
| [inference.md](./inference.md) | Inference 패키지 설계 |
| [api.md](./api.md) | API 패키지 설계 |
| [cli.md](./cli.md) | CLI 패키지 설계 |
| [rag.md](./rag.md) | RAG 패키지 설계 |
| [langchain.md](./langchain.md) | LangChain 패키지 설계 |

## 기술 스택

| 계층 | 기술 |
|------|------|
| Runtime | Python 3.10-3.12 |
| ML Framework | MLX (Apple Silicon) |
| Package Manager | Poetry (Workspace) |
| HTTP | FastAPI + Uvicorn |
| CLI | Click + Rich |
| Validation | Pydantic |
| Vector Store | FAISS |
| LangChain | langchain-core |
