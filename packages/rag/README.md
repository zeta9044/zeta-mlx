# zeta-mlx-rag

RAG (Retrieval-Augmented Generation) pipeline for Zeta MLX.

## Installation

```bash
pip install zeta-mlx-rag
```

## Features

- **Document Processing**: Text chunking and preprocessing
- **Embeddings**: Integration with zeta-mlx-embedding
- **Retrieval**: Vector similarity search
- **Generation**: Context-aware LLM responses

## Usage

```python
from zeta_mlx.rag import RAGPipeline

pipeline = RAGPipeline(
    embedding_model="bge-m3",
    llm_model="qwen3-8b"
)

# Add documents
pipeline.add_documents(["doc1.txt", "doc2.txt"])

# Query
response = pipeline.query("What is MLX?")
```

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
