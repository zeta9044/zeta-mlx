# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 작업할 때 참고하는 가이드입니다.

## 프로젝트 개요

MLX LLM Server는 MLX 프레임워크를 사용하여 Apple Silicon에서 LLM 모델을 실행하는 OpenAI 호환 추론 서버입니다. Qwen 모델(Qwen3 포함)을 커스텀 모델 등록 방식으로 지원하며, OpenAI의 chat completion 형식으로 응답합니다.

**핵심 제약사항:**
- **Apple Silicon 전용**: MLX 프레임워크는 M1/M2/M3/M4 칩이 탑재된 macOS에서만 동작
- **Python 3.10-3.12 전용**: MLX는 Python 3.13 이상을 지원하지 않음
- **기본 포트**: 9044 (ZetaLab 시그니처)
- **기본 모델**: `mlx-community/Qwen3-8B-4bit`

## 주요 명령어

### 개발
```bash
# 의존성 설치
poetry install

# 서버 실행 (기본값: Qwen3-8B-4bit, 포트 9044)
poetry run mlx-llm-server

# 커스텀 모델로 실행
poetry run mlx-llm-server --model-name mlx-community/Qwen2.5-7B-Instruct-4bit --port 9044

# 코드 포맷팅
poetry run black src tests
poetry run ruff check src tests
poetry run ruff check --fix src tests  # 자동 수정

# 테스트 실행
poetry run pytest
poetry run pytest -v -k "test_name"  # 단일 테스트 실행
```

### Python 버전 설정
```bash
# Python 3.12 사용 확인 (3.14 아님)
poetry env use python3.12
poetry env info  # 버전 확인
```

## 아키텍처

### 커스텀 모델 등록 시스템

이 프로젝트는 MLX-LM에 Qwen3 지원을 추가하기 위한 **동적 모델 등록 시스템**을 구현합니다:

1. **커스텀 모델 구현** (`src/mlx_llm_server/custom_models/qwen3.py`):
   - query/key 정규화 레이어를 포함한 Qwen3 아키텍처 구현
   - Qwen2와의 주요 차이점: attention에 `q_norm`과 `k_norm` RMSNorm 레이어 추가
   - attention projection 레이어에 bias 없음 (`q_proj`, `k_proj`, `v_proj`, `o_proj`)

2. **동적 등록** (`src/mlx_llm_server/model_loader.py`):
   - 런타임에 커스텀 모델을 `sys.modules['mlx_lm.models.qwen3']`에 등록
   - `inference.py`에서 모델 로딩 전에 호출
   - MLX-LM이 커스텀 모델 구현을 찾아 사용할 수 있도록 함

3. **추론 파이프라인** (`src/mlx_llm_server/inference.py`):
   - `setup_custom_models()`는 반드시 `mlx_lm.load()` 전에 호출해야 함
   - 채팅 템플릿 적용 및 토큰 생성 처리
   - 스트리밍 및 비스트리밍 생성 모두 지원

### OpenAI API 호환성

서버는 완전한 OpenAI API 호환성을 제공합니다 (`src/mlx_llm_server/app.py`):

**엔드포인트:**
- `GET /health` - 헬스 체크
- `GET /v1/models` - 사용 가능한 모델 목록
- `POST /v1/chat/completions` - 채팅 완성 (스트리밍 및 비스트리밍)

**응답 형식:**
- `id`: 고유 완성 ID (`chatcmpl-{uuid}`)
- `object`: "chat.completion" 또는 "chat.completion.chunk"
- `created`: Unix 타임스탬프
- `choices`: message와 finish_reason이 포함된 배열
- `usage`: 토큰 수 (prompt_tokens, completion_tokens, total_tokens)

**스트리밍 형식:**
- 첫 번째 청크: role 할당
- 콘텐츠 청크: 증분 텍스트
- 마지막 청크: finish_reason "stop"
- 종료자: `data: [DONE]`

**UTF-8 스트리밍:** `inference.py`의 스트리밍 구현은 토큰 누적과 대체 문자 감지를 사용하여 멀티바이트 문자(한국어, 중국어, 일본어)를 올바르게 처리합니다. 불완전한 UTF-8 시퀀스는 완성될 때까지 버퍼링됩니다.

### 프로젝트 구조

```
src/mlx_llm_server/
├── custom_models/      # 커스텀 MLX 모델 구현
│   ├── __init__.py
│   └── qwen3.py       # q_norm/k_norm 레이어가 있는 Qwen3
├── model_loader.py    # sys.modules에 동적 모델 등록
├── inference.py       # 커스텀 모델 설정이 포함된 MLX 추론 엔진
├── models.py          # OpenAI 호환 Pydantic 모델
├── app.py             # /v1/chat/completions가 있는 FastAPI 서버
├── cli.py             # CLI 진입점
└── config.py          # 설정 (기본 모델, 포트 9044)
```

## 새 커스텀 모델 추가

MLX-LM에 없는 모델 지원을 추가하려면:

1. **모델 구현** (`custom_models/{model_name}.py`):
   - `qwen3.py`를 템플릿으로 복사
   - `ModelArgs`, `Attention`, `MLP`, `TransformerBlock`, `Model` 구현
   - HuggingFace 모델 config의 정확한 weight 이름과 일치시킴

2. **모델 등록** (`model_loader.py`):
   ```python
   def register_new_model():
       from mlx_llm_server.custom_models import new_model
       sys.modules['mlx_lm.models.new_model'] = new_model

   def setup_custom_models():
       register_qwen3_model()
       register_new_model()  # 여기에 추가
   ```

3. **로딩 테스트**:
   ```bash
   poetry run mlx-llm-server --model-name mlx-community/NewModel-4bit
   ```

## 중요 사항

- **Python 3.13+ 절대 사용 금지**: MLX와 호환되지 않음
- **토큰 카운팅**: `tokenizer.encode()` 길이 사용 (정확한 OpenAI 수치와 다를 수 있음)
- **포트 9044**: ZetaLab 시그니처 포트, 기본값 변경 금지
- **모델 이름**: HuggingFace config.json의 `model_type` 필드와 일치해야 함
- **OpenAI SDK 사용**: `base_url="http://localhost:9044/v1"` 설정, `api_key`는 아무 값이나 사용
