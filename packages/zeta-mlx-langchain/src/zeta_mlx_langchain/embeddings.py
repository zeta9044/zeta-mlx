"""LangChain Embeddings 통합"""
from typing import List
from langchain_core.embeddings import Embeddings
from zeta_mlx_core import Result, Failure


class MLXLLMEmbeddings(Embeddings):
    """LangChain Embeddings for MLX LLM

    간단한 해시 기반 임베딩 또는 SentenceTransformer 사용

    Example:
        from zeta_mlx_langchain import MLXLLMEmbeddings

        embeddings = MLXLLMEmbeddings()
        vectors = embeddings.embed_documents(["Hello", "World"])
    """

    model_name: str = "all-MiniLM-L6-v2"
    use_sentence_transformers: bool = False
    dimension: int = 384

    _embed_fn: any = None

    class Config:
        arbitrary_types_allowed = True

    def _ensure_embed_fn(self) -> None:
        """임베딩 함수 초기화"""
        if self._embed_fn is not None:
            return

        if self.use_sentence_transformers:
            from zeta_mlx_rag import create_sentence_transformer_fn
            result = create_sentence_transformer_fn(self.model_name)
            if isinstance(result, Failure):
                raise RuntimeError(f"Failed to load embedding model: {result.error}")
            self._embed_fn = result.value
        else:
            from zeta_mlx_rag import create_simple_embedding_fn
            self._embed_fn = create_simple_embedding_fn(self.dimension)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩"""
        self._ensure_embed_fn()
        embeddings = self._embed_fn(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩"""
        self._ensure_embed_fn()
        embedding = self._embed_fn([text])[0]
        return embedding.tolist()
