"""에러 타입 정의 (OR Type)"""
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ValidationError:
    """검증 에러"""
    field: str
    message: str
    code: str = "VALIDATION_ERROR"


@dataclass(frozen=True)
class TokenLimitError:
    """토큰 제한 초과"""
    limit: int
    actual: int
    code: str = "TOKEN_LIMIT_EXCEEDED"


@dataclass(frozen=True)
class ModelNotFoundError:
    """모델 없음"""
    model: str
    code: str = "MODEL_NOT_FOUND"


@dataclass(frozen=True)
class GenerationError:
    """생성 에러"""
    reason: str
    code: str = "GENERATION_ERROR"


# OR Type: 모든 추론 에러
InferenceError = Union[
    ValidationError,
    TokenLimitError,
    ModelNotFoundError,
    GenerationError,
]


def error_to_dict(error: InferenceError) -> dict:
    """에러를 딕셔너리로 변환 (API 응답용)"""
    match error:
        case ValidationError(field, message, code):
            return {"code": code, "field": field, "message": message}
        case TokenLimitError(limit, actual, code):
            return {"code": code, "limit": limit, "actual": actual, "message": f"Token limit exceeded: {actual} > {limit}"}
        case ModelNotFoundError(model, code):
            return {"code": code, "model": model, "message": f"Model not found: {model}"}
        case GenerationError(reason, code):
            return {"code": code, "reason": reason, "message": reason}
