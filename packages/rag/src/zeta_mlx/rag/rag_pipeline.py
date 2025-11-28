"""RAG 파이프라인 (Railway 패턴)"""
from typing import Callable, Sequence
from dataclasses import dataclass
from zeta_mlx.core import (
    Result, Success, Failure,
    Message, GenerationParams, ChatRequest,
    GenerationResult, InferenceError,
    pipe,
)
from zeta_mlx.rag.types import Document, SearchResult, RAGContext, EmbeddingFn
from zeta_mlx.rag.vector_store import InMemoryVectorStore, vector_store_from_documents
from zeta_mlx.rag.retriever import create_retriever


@dataclass(frozen=True)
class RAGConfig:
    """RAG 설정"""
    top_k: int = 3
    min_score: float = 0.0
    system_template: str = """You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer based on the context above. If the context doesn't contain relevant information, say so."""


@dataclass(frozen=True)
class RAGRequest:
    """RAG 요청"""
    query: str
    config: RAGConfig


@dataclass(frozen=True)
class RAGResponse:
    """RAG 응답"""
    answer: str
    context: RAGContext
    generation_result: GenerationResult


# ============================================================
# RAG 파이프라인 빌더
# ============================================================

class RAGPipeline:
    """RAG 파이프라인

    검색 → 컨텍스트 구성 → LLM 생성
    """

    def __init__(
        self,
        retriever: Callable[[str, int], list[SearchResult]],
        generate_fn: Callable[[ChatRequest], Result[GenerationResult, InferenceError]],
        config: RAGConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._generate = generate_fn
        self._config = config or RAGConfig()

    def run(self, query: str) -> Result[RAGResponse, InferenceError]:
        """RAG 파이프라인 실행"""
        # 1. 검색
        results = self._retriever(query, self._config.top_k)

        # 2. 점수 필터링
        filtered = [r for r in results if r.score >= self._config.min_score]

        # 3. 컨텍스트 생성
        context = RAGContext(query=query, results=tuple(filtered))
        context_str = context.to_context_string()

        # 4. 프롬프트 구성
        system_message = self._config.system_template.format(
            context=context_str,
            question=query,
        )

        messages = [
            Message(role="system", content=system_message),
            Message(role="user", content=query),
        ]

        request = ChatRequest(
            model="",  # 엔진에서 처리
            messages=messages,
            params=GenerationParams(),
            stream=False,
        )

        # 5. 생성
        result = self._generate(request)

        if isinstance(result, Failure):
            return result

        return Success(RAGResponse(
            answer=result.value.text,
            context=context,
            generation_result=result.value,
        ))

    def __call__(self, query: str) -> Result[RAGResponse, InferenceError]:
        """함수형 인터페이스"""
        return self.run(query)


# ============================================================
# 팩토리 함수
# ============================================================

def create_rag_pipeline(
    documents: Sequence[Document],
    embed_fn: EmbeddingFn,
    generate_fn: Callable[[ChatRequest], Result[GenerationResult, InferenceError]],
    config: RAGConfig | None = None,
) -> RAGPipeline:
    """RAG 파이프라인 생성

    Example:
        documents = [Document(id="1", content="..."), ...]
        embed_fn = create_simple_embedding_fn()
        generate_fn = engine.generate

        rag = create_rag_pipeline(documents, embed_fn, generate_fn)
        result = rag("What is the meaning of life?")
    """
    # 임베딩 생성
    contents = [doc.content for doc in documents]
    embeddings = embed_fn(contents)

    # 벡터 저장소 생성
    store = vector_store_from_documents(documents, embeddings)

    # 검색기 생성
    retriever = create_retriever(embed_fn, store)

    return RAGPipeline(retriever, generate_fn, config)


def create_rag_from_texts(
    texts: Sequence[str],
    embed_fn: EmbeddingFn,
    generate_fn: Callable[[ChatRequest], Result[GenerationResult, InferenceError]],
    config: RAGConfig | None = None,
) -> RAGPipeline:
    """텍스트 목록에서 RAG 파이프라인 생성"""
    documents = [
        Document(id=f"doc_{i}", content=text)
        for i, text in enumerate(texts)
    ]
    return create_rag_pipeline(documents, embed_fn, generate_fn, config)


# ============================================================
# 문서 처리 유틸리티
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """텍스트를 청크로 분할"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def documents_from_texts(
    texts: Sequence[str],
    metadata: dict | None = None,
) -> list[Document]:
    """텍스트 목록을 Document 목록으로 변환"""
    return [
        Document(id=f"doc_{i}", content=text, metadata=metadata)
        for i, text in enumerate(texts)
    ]
