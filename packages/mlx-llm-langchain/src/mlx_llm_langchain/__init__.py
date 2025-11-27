"""MLX LLM LangChain Integration"""
from mlx_llm_langchain.chat_model import ChatMLXLLM
from mlx_llm_langchain.embeddings import MLXLLMEmbeddings
from mlx_llm_langchain.tools import (
    ToolExecutor,
    create_tool_executor,
    create_tool_prompt,
    parse_tool_response,
)

__version__ = "0.1.0"

__all__ = [
    # Chat Model
    "ChatMLXLLM",
    # Embeddings
    "MLXLLMEmbeddings",
    # Tools
    "ToolExecutor",
    "create_tool_executor",
    "create_tool_prompt",
    "parse_tool_response",
]
