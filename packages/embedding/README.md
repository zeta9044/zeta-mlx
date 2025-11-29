# zeta-mlx-embedding

Embedding model serving for Zeta MLX platform.

## Installation

```bash
pip install zeta-mlx-embedding
```

## Features

- **OpenAI Compatible**: `/v1/embeddings` endpoint
- **Multilingual**: BGE-M3, E5-Large support
- **High Dimension**: 1024-dimensional embeddings

## Supported Models

| Model | Dimension | Description |
|-------|-----------|-------------|
| bge-m3 | 1024 | Korean/English optimized |
| multilingual-e5-large | 1024 | Multilingual |
| all-MiniLM-L6-v2 | 384 | Lightweight |

## Usage

```python
from zeta_mlx.embedding import EmbeddingEngine

engine = EmbeddingEngine("BAAI/bge-m3")
embeddings = engine.embed(["Hello", "안녕하세요"])
```

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
