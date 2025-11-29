"""토큰화 API 라우트 (vLLM 호환)"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["tokenize"])

_model_manager = None


def set_model_manager(manager) -> None:
    global _model_manager
    _model_manager = manager


class TokenizeRequest(BaseModel):
    """토큰화 요청 (vLLM 호환)"""
    model: str | None = None
    prompt: str
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    """토큰화 응답 (vLLM 호환)"""
    tokens: list[int]
    count: int
    max_model_len: int


class DetokenizeRequest(BaseModel):
    """디토큰화 요청 (vLLM 호환)"""
    model: str | None = None
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    """디토큰화 응답 (vLLM 호환)"""
    prompt: str


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest) -> TokenizeResponse:
    """텍스트를 토큰으로 변환 (vLLM 호환)"""
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    from zeta_mlx.core import Failure

    model_alias = request.model or _model_manager.default_alias
    model_alias = _model_manager.resolve_alias(model_alias)

    engine_result = _model_manager.get_engine(model_alias)
    if isinstance(engine_result, Failure):
        raise HTTPException(status_code=404, detail=f"Model '{model_alias}' not found")

    engine = engine_result.value

    try:
        tokenizer = engine._bundle.tokenizer
        tokens = tokenizer.encode(request.prompt, add_special_tokens=request.add_special_tokens)
        max_model_len = getattr(engine._bundle.model.args, 'max_position_embeddings', 32768)

        return TokenizeResponse(
            tokens=tokens if isinstance(tokens, list) else tokens.tolist(),
            count=len(tokens),
            max_model_len=max_model_len,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")


@router.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(request: DetokenizeRequest) -> DetokenizeResponse:
    """토큰을 텍스트로 변환 (vLLM 호환)"""
    if _model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")

    from zeta_mlx.core import Failure

    model_alias = request.model or _model_manager.default_alias
    model_alias = _model_manager.resolve_alias(model_alias)

    engine_result = _model_manager.get_engine(model_alias)
    if isinstance(engine_result, Failure):
        raise HTTPException(status_code=404, detail=f"Model '{model_alias}' not found")

    engine = engine_result.value

    try:
        tokenizer = engine._bundle.tokenizer
        prompt = tokenizer.decode(request.tokens)
        return DetokenizeResponse(prompt=prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detokenization failed: {str(e)}")
