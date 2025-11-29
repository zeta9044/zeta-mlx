# Zeta MLX

Apple Silicon에서 MLX 기반 LLM/임베딩 추론을 위한 OpenAI 호환 플랫폼

## 특징

- **OpenAI 호환 API**: `/v1/chat/completions`, `/v1/embeddings` 엔드포인트 제공
- **vLLM 호환 API**: `/tokenize`, `/detokenize` 엔드포인트 제공
- **Apple Silicon 최적화**: MLX 프레임워크 기반 네이티브 성능
- **단일 모델 로딩**: 메모리 효율적인 모델 관리 (서버당 1개 모델)
- **다국어 임베딩**: BGE-M3 등 한/영 혼용 지원 (1024 차원)
- **모노레포 구조**: 독립적인 패키지로 모듈화

## 패키지 구조

```
zeta-mlx/
├── packages/
│   ├── core/        # 공통 타입, Result 모나드, 설정
│   ├── cli/         # Typer 기반 CLI
│   ├── inference/   # MLX LLM 추론 엔진 + FastAPI 서버
│   ├── embedding/   # 임베딩 엔진 + FastAPI 서버
│   ├── rag/         # RAG 파이프라인
│   └── langchain/   # LangChain 통합
└── config.yaml      # 모델 설정
```

## 요구사항

- Python 3.10 ~ 3.12
- Apple Silicon Mac (M1/M2/M3/M4)
- Poetry 1.8+

## 설치

```bash
# 저장소 클론
git clone https://github.com/zeta9044/zeta-mlx.git
cd zeta-mlx

# 의존성 설치
poetry install
```

## 사용법

### CLI 실행

```bash
# 도움말
poetry run python -m zeta_mlx.cli --help

# 버전 확인
poetry run python -m zeta_mlx.cli version
```

### LLM 서버 (포트 9044)

```bash
# 서버 시작
poetry run python -m zeta_mlx.cli llm start

# 특정 모델로 시작
poetry run python -m zeta_mlx.cli llm start -m qwen3-8b

# 데몬 모드
poetry run python -m zeta_mlx.cli llm start --daemon
```

### 임베딩 서버 (포트 9045)

```bash
# 서버 시작
poetry run python -m zeta_mlx.cli embedding start

# 특정 모델로 시작
poetry run python -m zeta_mlx.cli embedding start -m multilingual-e5-large
```

### API 사용 예시

**LLM Chat Completions:**
```bash
curl -X POST http://localhost:9044/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Embeddings:**
```bash
curl -X POST http://localhost:9045/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["안녕하세요", "Hello"]
  }'
```

**Tokenize (vLLM 호환):**
```bash
curl -X POST http://localhost:9044/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "add_special_tokens": true
  }'
```

**Detokenize (vLLM 호환):**
```bash
curl -X POST http://localhost:9044/detokenize \
  -H "Content-Type: application/json" \
  -d '{
    "tokens": [9707, 11, 1917, 0]
  }'
```

**Health Check:**
```bash
curl http://localhost:9044/health  # LLM
curl http://localhost:9045/health  # Embedding
```

## 설정

`config.yaml`에서 모델 설정:

```yaml
models:
  default: qwen3-4b
  available:
    qwen3-4b:
      path: mlx-community/Qwen3-4B-4bit
      description: "Qwen3 4B - 경량 모델"

embedding_models:
  default: bge-m3
  available:
    bge-m3:
      path: BAAI/bge-m3
      provider: sentence-transformers
      dimension: 1024
      description: "BGE-M3 - 한/영 혼용 최적"
```

## 지원 모델

### LLM 모델
| 별칭 | 모델 | 크기 | 용도 |
|------|------|------|------|
| qwen3-8b | mlx-community/Qwen3-8B-4bit | 8GB | 기본 추천 |
| qwen3-4b | mlx-community/Qwen3-4B-4bit | 4GB | 경량 |
| qwen2.5-7b | mlx-community/Qwen2.5-7B-Instruct-4bit | 7GB | Instruction |
| llama3.2-3b | mlx-community/Llama-3.2-3B-Instruct-4bit | 3GB | 경량 |

### 임베딩 모델
| 별칭 | 모델 | 차원 | 특징 |
|------|------|------|------|
| bge-m3 | BAAI/bge-m3 | 1024 | 한/영 혼용 최적 |
| multilingual-e5-large | intfloat/multilingual-e5-large | 1024 | 다국어 |
| minilm | sentence-transformers/all-MiniLM-L6-v2 | 384 | 경량/빠름 |

## 개발

```bash
# 테스트 실행
poetry run pytest

# 린트
poetry run ruff check .

# 타입 체크
poetry run mypy .
```

## 라이선스

MIT License
