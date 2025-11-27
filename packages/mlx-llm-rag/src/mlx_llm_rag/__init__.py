"""MLX LLM RAG - Retrieval-Augmented Generation"""
from mlx_llm_rag.types import (
    Document, Embedding, SearchResult, RAGContext,
    EmbeddingFn, VectorStoreFn, RetrieverFn,
)
from mlx_llm_rag.embeddings import (
    create_simple_embedding_fn,
    create_sentence_transformer_fn,
    create_embedding_fn_from_config,
    batch_embed,
    cosine_similarity,
    cosine_similarity_batch,
)
from mlx_llm_rag.vector_store import (
    InMemoryVectorStore,
    create_vector_store,
    vector_store_from_documents,
)
from mlx_llm_rag.retriever import (
    Retriever,
    create_retriever,
    filter_by_score,
    filter_by_metadata,
    rerank_by_length,
)
from mlx_llm_rag.rag_pipeline import (
    RAGConfig, RAGRequest, RAGResponse,
    RAGPipeline,
    create_rag_pipeline,
    create_rag_from_texts,
    chunk_text,
    documents_from_texts,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Document",
    "Embedding",
    "SearchResult",
    "RAGContext",
    "EmbeddingFn",
    "VectorStoreFn",
    "RetrieverFn",
    # Embeddings
    "create_simple_embedding_fn",
    "create_sentence_transformer_fn",
    "create_embedding_fn_from_config",
    "batch_embed",
    "cosine_similarity",
    "cosine_similarity_batch",
    # Vector Store
    "InMemoryVectorStore",
    "create_vector_store",
    "vector_store_from_documents",
    # Retriever
    "Retriever",
    "create_retriever",
    "filter_by_score",
    "filter_by_metadata",
    "rerank_by_length",
    # RAG Pipeline
    "RAGConfig",
    "RAGRequest",
    "RAGResponse",
    "RAGPipeline",
    "create_rag_pipeline",
    "create_rag_from_texts",
    "chunk_text",
    "documents_from_texts",
]
