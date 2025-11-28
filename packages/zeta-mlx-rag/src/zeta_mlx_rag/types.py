"""RAG 도메인 타입"""
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np
from numpy.typing import NDArray


# ============================================================
# 기본 타입
# ============================================================

@dataclass(frozen=True)
class Document:
    """검색 문서"""
    id: str
    content: str
    metadata: dict | None = None


@dataclass(frozen=True)
class Embedding:
    """임베딩 벡터"""
    vector: NDArray[np.float32]
    document_id: str


@dataclass(frozen=True)
class SearchResult:
    """검색 결과"""
    document: Document
    score: float
    rank: int


@dataclass(frozen=True)
class RAGContext:
    """RAG 컨텍스트 (검색된 문서들)"""
    query: str
    results: tuple[SearchResult, ...]

    def to_context_string(self, max_docs: int = 3) -> str:
        """컨텍스트 문자열로 변환"""
        docs = self.results[:max_docs]
        context_parts = []
        for i, result in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]\n{result.document.content}")
        return "\n\n".join(context_parts)


# ============================================================
# 프로토콜 (인터페이스)
# ============================================================

class EmbeddingFn(Protocol):
    """임베딩 함수 프로토콜"""
    def __call__(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """텍스트들을 임베딩 벡터로 변환"""
        ...


class VectorStoreFn(Protocol):
    """벡터 저장소 프로토콜"""
    def add(self, embeddings: Sequence[Embedding]) -> None:
        """임베딩 추가"""
        ...

    def search(self, query_embedding: NDArray[np.float32], top_k: int) -> list[tuple[str, float]]:
        """유사 문서 검색 (document_id, score)"""
        ...


class RetrieverFn(Protocol):
    """검색기 프로토콜"""
    def __call__(self, query: str, top_k: int) -> list[SearchResult]:
        """쿼리로 문서 검색"""
        ...
