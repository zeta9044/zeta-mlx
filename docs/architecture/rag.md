# RAG 패키지 (packages/rag)

Retrieval-Augmented Generation 파이프라인입니다.

## 모듈 구조

```
zeta_mlx/rag/
├── __init__.py
├── types.py          # RAG 도메인 타입
├── chunker.py        # 문서 청킹
├── embedder.py       # 임베딩 생성
├── retriever.py      # 검색기
├── pipeline.py       # RAG 파이프라인
└── vectorstore/      # 벡터 저장소
    ├── __init__.py
    ├── protocol.py   # Protocol 정의
    └── faiss.py      # FAISS 구현
```

## types.py - RAG 도메인 타입

```python
"""RAG 도메인 타입"""
from dataclasses import dataclass
from typing import NewType

ChunkId = NewType('ChunkId', str)
CollectionName = NewType('CollectionName', str)


@dataclass(frozen=True)
class Document:
    """원본 문서"""
    id: str
    content: str
    metadata: dict


@dataclass(frozen=True)
class Chunk:
    """문서 청크"""
    id: ChunkId
    document_id: str
    content: str
    start_index: int
    end_index: int
    metadata: dict


@dataclass(frozen=True)
class Embedding:
    """임베딩 벡터"""
    chunk_id: ChunkId
    vector: tuple[float, ...]  # 불변 리스트


@dataclass(frozen=True)
class SearchResult:
    """검색 결과"""
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RAGContext:
    """RAG 컨텍스트"""
    query: str
    results: tuple[SearchResult, ...]
    formatted: str
```

## chunker.py - 문서 청킹

```python
"""문서 청킹"""
from typing import Iterator
import uuid

from zeta_mlx.rag.types import Document, Chunk, ChunkId


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> Iterator[Chunk]:
    """
    문서를 청크로 분할 (Generator)

    순수 함수: 같은 입력 → 같은 출력
    """
    content = document.content
    start = 0
    chunk_index = 0

    while start < len(content):
        # 청크 끝 위치
        end = start + chunk_size

        # separator로 자연스러운 분할점 찾기
        if end < len(content):
            split_pos = content.rfind(separator, start, end)
            if split_pos > start:
                end = split_pos + len(separator)

        chunk_content = content[start:end].strip()

        if chunk_content:
            yield Chunk(
                id=ChunkId(f"{document.id}_{chunk_index}"),
                document_id=document.id,
                content=chunk_content,
                start_index=start,
                end_index=end,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                },
            )
            chunk_index += 1

        # 다음 시작 (overlap 고려)
        start = end - chunk_overlap if end < len(content) else end


def chunk_documents(
    documents: Iterator[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Iterator[Chunk]:
    """여러 문서 청킹"""
    for doc in documents:
        yield from chunk_document(doc, chunk_size, chunk_overlap)
```

## embedder.py - 임베딩 생성

```python
"""임베딩 생성"""
from typing import Iterator, Callable
from functools import lru_cache

from zeta_mlx.rag.types import Chunk, Embedding


# 임베딩 함수 타입
EmbedFn = Callable[[list[str]], list[list[float]]]


def create_sentence_transformer_embedder(
    model_name: str = "all-MiniLM-L6-v2"
) -> EmbedFn:
    """SentenceTransformer 임베딩 함수 생성"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return embed


def embed_chunks(
    chunks: list[Chunk],
    embed_fn: EmbedFn,
    batch_size: int = 32,
) -> Iterator[Embedding]:
    """
    청크 임베딩 생성

    배치 처리로 효율성 향상
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.content for c in batch]
        vectors = embed_fn(texts)

        for chunk, vector in zip(batch, vectors):
            yield Embedding(
                chunk_id=chunk.id,
                vector=tuple(vector),
            )


def embed_query(query: str, embed_fn: EmbedFn) -> tuple[float, ...]:
    """쿼리 임베딩"""
    vectors = embed_fn([query])
    return tuple(vectors[0])
```

## vectorstore/protocol.py - 벡터 저장소 Protocol

```python
"""벡터 저장소 Protocol"""
from typing import Protocol, Iterator

from zeta_mlx.rag.types import Chunk, Embedding, SearchResult, CollectionName


class VectorStore(Protocol):
    """벡터 저장소 인터페이스"""

    def add(
        self,
        collection: CollectionName,
        chunks: list[Chunk],
        embeddings: list[Embedding],
    ) -> None:
        """청크와 임베딩 추가"""
        ...

    def search(
        self,
        collection: CollectionName,
        query_embedding: tuple[float, ...],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """유사도 검색"""
        ...

    def delete_collection(self, collection: CollectionName) -> None:
        """컬렉션 삭제"""
        ...

    def list_collections(self) -> list[CollectionName]:
        """컬렉션 목록"""
        ...
```

## pipeline.py - RAG 파이프라인

```python
"""RAG 파이프라인"""
from typing import Callable

from zeta_mlx.core import (
    Result, Success, Failure, Railway,
    Message, NonEmptyList,
)
from zeta_mlx.rag.types import (
    Document, Chunk, SearchResult, RAGContext, CollectionName,
)
from zeta_mlx.rag.chunker import chunk_documents
from zeta_mlx.rag.embedder import embed_chunks, embed_query, EmbedFn
from zeta_mlx.rag.vectorstore.protocol import VectorStore


# ============================================================
# 인덱싱 파이프라인
# ============================================================

def create_index_pipeline(
    vectorstore: VectorStore,
    embed_fn: EmbedFn,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Callable[[CollectionName, list[Document]], Result[int, str]]:
    """인덱싱 파이프라인 생성"""

    def index(
        collection: CollectionName,
        documents: list[Document],
    ) -> Result[int, str]:
        try:
            # 청킹
            chunks = list(chunk_documents(iter(documents), chunk_size, chunk_overlap))

            if not chunks:
                return Failure("No chunks created")

            # 임베딩
            embeddings = list(embed_chunks(chunks, embed_fn))

            # 저장
            vectorstore.add(collection, chunks, embeddings)

            return Success(len(chunks))
        except Exception as e:
            return Failure(str(e))

    return index


# ============================================================
# 검색 파이프라인
# ============================================================

def create_retrieve_pipeline(
    vectorstore: VectorStore,
    embed_fn: EmbedFn,
    top_k: int = 5,
) -> Callable[[CollectionName, str], Result[RAGContext, str]]:
    """검색 파이프라인 생성"""

    def retrieve(
        collection: CollectionName,
        query: str,
    ) -> Result[RAGContext, str]:
        try:
            # 쿼리 임베딩
            query_embedding = embed_query(query, embed_fn)

            # 검색
            results = vectorstore.search(collection, query_embedding, top_k)

            if not results:
                return Failure("No results found")

            # 컨텍스트 포맷팅
            formatted = _format_context(results)

            return Success(RAGContext(
                query=query,
                results=tuple(results),
                formatted=formatted,
            ))
        except Exception as e:
            return Failure(str(e))

    return retrieve


def _format_context(results: list[SearchResult]) -> str:
    """검색 결과를 컨텍스트 문자열로 포맷"""
    parts = []
    for i, result in enumerate(results, 1):
        parts.append(f"[{i}] {result.chunk.content}")
    return "\n\n".join(parts)


# ============================================================
# RAG 생성 파이프라인
# ============================================================

def create_rag_prompt(context: RAGContext, question: str) -> list[Message]:
    """RAG 프롬프트 생성"""
    system = f"""다음 컨텍스트를 참고하여 질문에 답하세요.

컨텍스트:
{context.formatted}

규칙:
1. 컨텍스트에 없는 내용은 추측하지 마세요
2. 답변의 근거를 컨텍스트에서 찾아 언급하세요
"""

    return [
        Message(role="system", content=system),
        Message(role="user", content=question),
    ]
```

## Public API (__init__.py)

```python
"""Zeta MLX RAG - Retrieval-Augmented Generation"""
from zeta_mlx.rag.types import (
    Document, Chunk, Embedding, SearchResult, RAGContext,
    ChunkId, CollectionName,
)
from zeta_mlx.rag.chunker import chunk_document, chunk_documents
from zeta_mlx.rag.embedder import (
    embed_chunks, embed_query,
    create_sentence_transformer_embedder,
    EmbedFn,
)
from zeta_mlx.rag.vectorstore.protocol import VectorStore
from zeta_mlx.rag.pipeline import (
    create_index_pipeline,
    create_retrieve_pipeline,
    create_rag_prompt,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Document", "Chunk", "Embedding", "SearchResult", "RAGContext",
    "ChunkId", "CollectionName",
    # Chunker
    "chunk_document", "chunk_documents",
    # Embedder
    "embed_chunks", "embed_query",
    "create_sentence_transformer_embedder", "EmbedFn",
    # VectorStore
    "VectorStore",
    # Pipeline
    "create_index_pipeline", "create_retrieve_pipeline",
    "create_rag_prompt",
]
```
