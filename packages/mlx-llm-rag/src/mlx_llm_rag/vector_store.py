"""벡터 저장소 (순수 함수형)"""
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from mlx_llm_rag.types import Document, Embedding
from mlx_llm_rag.embeddings import cosine_similarity_batch


@dataclass
class InMemoryVectorStore:
    """인메모리 벡터 저장소

    함수형 스타일: 상태 변경 시 새 인스턴스 반환 가능
    실용적 이유로 가변 구현 (대량 문서 처리 시 성능)
    """
    documents: dict[str, Document] = field(default_factory=dict)
    embeddings: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    _matrix: NDArray[np.float32] | None = field(default=None, repr=False)
    _doc_ids: list[str] = field(default_factory=list, repr=False)

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: NDArray[np.float32],
    ) -> None:
        """문서와 임베딩 추가"""
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")

        for doc, emb in zip(documents, embeddings):
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = emb

        # 검색용 매트릭스 재구성
        self._rebuild_matrix()

    def add_embedding(self, embedding: Embedding, document: Document) -> None:
        """단일 임베딩 추가"""
        self.documents[document.id] = document
        self.embeddings[document.id] = embedding.vector
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """검색용 매트릭스 재구성"""
        if not self.embeddings:
            self._matrix = None
            self._doc_ids = []
            return

        self._doc_ids = list(self.embeddings.keys())
        self._matrix = np.array([self.embeddings[did] for did in self._doc_ids])

    def search(
        self,
        query_embedding: NDArray[np.float32],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """유사 문서 검색

        Returns:
            List of (document_id, similarity_score) sorted by score descending
        """
        if self._matrix is None or len(self._matrix) == 0:
            return []

        # 코사인 유사도 계산
        similarities = cosine_similarity_batch(query_embedding, self._matrix)

        # Top-K 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_id = self._doc_ids[idx]
            score = float(similarities[idx])
            results.append((doc_id, score))

        return results

    def get_document(self, doc_id: str) -> Document | None:
        """문서 ID로 문서 조회"""
        return self.documents.get(doc_id)

    def __len__(self) -> int:
        return len(self.documents)


# ============================================================
# 팩토리 함수
# ============================================================

def create_vector_store() -> InMemoryVectorStore:
    """새 벡터 저장소 생성"""
    return InMemoryVectorStore()


def vector_store_from_documents(
    documents: Sequence[Document],
    embeddings: NDArray[np.float32],
) -> InMemoryVectorStore:
    """문서와 임베딩으로 벡터 저장소 생성"""
    store = InMemoryVectorStore()
    store.add_documents(documents, embeddings)
    return store
