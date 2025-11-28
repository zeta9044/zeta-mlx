"""토큰화 API 라우트 (vLLM 호환)"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["tokenize"])

# 모델 매니저 (앱 시작시 주입)
_model_manager = None


def set_model_manager(manager) -> None:
    """모델 매니저 설정"""
    global _model_manager
    _model_manager = manager


# ============================================================
# Request/Response DTOs
# ============================================================

class TokenizeRequest(BaseModel):
    """토큰화 요청 (vLLM 호환)"""
    model: str
    prompt: str
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    """토큰화 응답 (vLLM 호환)"""
    tokens: list[int]
    count: int
    max_model_len: int


class DetokenizeRequest(BaseModel):
    """디토큰화 요청 (vLLM 호환)"""
    model: str
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    """디토큰화 응답 (vLLM 호환)"""
    prompt: str


# ============================================================
# Routes
# ============================================================

@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    """
    텍스트를 토큰으로 변환 (vLLM 호환)

    - model: 모델 이름 또는 별칭
    - prompt: 토큰화할 텍스트
    - add_special_tokens: 특수 토큰 추가 여부 (기본값: true)
    """
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    # 모델 별칭 해석
    model_alias = _model_manager.resolve_alias(request.model)

    # 엔진 가져오기
    from zeta_mlx.core import Failure
    engine_result = _model_manager.get_engine(model_alias)

    if isinstance(engine_result, Failure):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_alias}' not found"
        )

    engine = engine_result.value

    # 토큰화
    try:
        tokenizer = engine._bundle.tokenizer

        if add_special_tokens := request.add_special_tokens:
            tokens = tokenizer.encode(request.prompt, add_special_tokens=True)
        else:
            tokens = tokenizer.encode(request.prompt, add_special_tokens=False)

        # max_model_len 추정 (모델 config에서 가져오거나 기본값)
        max_model_len = getattr(engine._bundle.model.args, 'max_position_embeddings', 8192)

        return TokenizeResponse(
            tokens=tokens if isinstance(tokens, list) else tokens.tolist(),
            count=len(tokens),
            max_model_len=max_model_len,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")


@router.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(request: DetokenizeRequest) -> DetokenizeResponse:
    """
    토큰을 텍스트로 변환 (vLLM 호환)

    - model: 모델 이름 또는 별칭
    - tokens: 디토큰화할 토큰 ID 리스트
    """
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    # 모델 별칭 해석
    model_alias = _model_manager.resolve_alias(request.model)

    # 엔진 가져오기
    from zeta_mlx.core import Failure
    engine_result = _model_manager.get_engine(model_alias)

    if isinstance(engine_result, Failure):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_alias}' not found"
        )

    engine = engine_result.value

    # 디토큰화
    try:
        tokenizer = engine._bundle.tokenizer
        prompt = tokenizer.decode(request.tokens)

        return DetokenizeResponse(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detokenization failed: {str(e)}")
