"""API 요청 DTO (외부 세계 - Pydantic)"""
from pydantic import BaseModel, Field


class EmbeddingRequestDTO(BaseModel):
    """OpenAI 호환 임베딩 요청"""
    model: str = Field(..., description="모델 이름")
    input: str | list[str] = Field(..., description="임베딩할 텍스트 (단일 또는 배열)")
    encoding_format: str = Field(default="float", description="인코딩 포맷 (float, base64)")

    def to_input_list(self) -> list[str]:
        """입력을 리스트로 정규화"""
        if isinstance(self.input, str):
            return [self.input]
        return self.input
