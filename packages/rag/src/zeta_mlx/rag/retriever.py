"""검색기 (Retriever)"""
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from zeta_mlx.rag.types import Document, SearchResult, EmbeddingFn
from zeta_mlx.rag.vector_store import InMemoryVectorStore


class Retriever:
    """문서 검색기

    임베딩 함수와 벡터 저장소를 조합하여 검색 수행
    """

    def __init__(
        self,
        embed_fn: EmbeddingFn,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._embed_fn = embed_fn
        self._store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """쿼리로 문서 검색"""
        # 쿼리 임베딩 생성
        query_embedding = self._embed_fn([query])[0]

        # 벡터 저장소에서 검색
        results = self._store.search(query_embedding, top_k)

        # SearchResult로 변환
        search_results = []
        for rank, (doc_id, score) in enumerate(results):
            document = self._store.get_document(doc_id)
            if document:
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    rank=rank,
                ))

        return search_results

    def __call__(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """함수형 인터페이스"""
        return self.search(query, top_k)


# ============================================================
# 함수형 검색기 팩토리
# ============================================================

def create_retriever(
    embed_fn: EmbeddingFn,
    vector_store: InMemoryVectorStore,
) -> Callable[[str, int], list[SearchResult]]:
    """검색 함수 생성 (클로저)

    Example:
        retrieve = create_retriever(embed_fn, store)
        results = retrieve("search query", top_k=5)
    """
    def retrieve(query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = embed_fn([query])[0]
        raw_results = vector_store.search(query_embedding, top_k)

        search_results = []
        for rank, (doc_id, score) in enumerate(raw_results):
            document = vector_store.get_document(doc_id)
            if document:
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    rank=rank,
                ))

        return search_results

    return retrieve


# ============================================================
# 검색 유틸리티
# ============================================================

def filter_by_score(
    results: list[SearchResult],
    min_score: float,
) -> list[SearchResult]:
    """최소 점수 이상인 결과만 필터링"""
    return [r for r in results if r.score >= min_score]


def filter_by_metadata(
    results: list[SearchResult],
    key: str,
    value: str,
) -> list[SearchResult]:
    """메타데이터로 필터링"""
    return [
        r for r in results
        if r.document.metadata and r.document.metadata.get(key) == value
    ]


def rerank_by_length(
    results: list[SearchResult],
    prefer_longer: bool = True,
) -> list[SearchResult]:
    """문서 길이로 재정렬 (간단한 재랭킹)"""
    sorted_results = sorted(
        results,
        key=lambda r: len(r.document.content),
        reverse=prefer_longer,
    )
    # rank 재할당
    return [
        SearchResult(
            document=r.document,
            score=r.score,
            rank=i,
        )
        for i, r in enumerate(sorted_results)
    ]
