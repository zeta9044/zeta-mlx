# zeta-mlx-inference

MLX-based LLM inference engine with OpenAI/vLLM compatible API.

## Installation

```bash
pip install zeta-mlx-inference
```

## Features

- **MLX Optimized**: Native Apple Silicon performance
- **OpenAI Compatible**: `/v1/chat/completions` endpoint
- **vLLM Compatible**: `/tokenize`, `/detokenize` endpoints
- **Streaming**: Real-time token streaming support
- **Multi-model**: Support for Qwen, Llama, and more

## Usage

```python
from zeta_mlx.inference import InferenceEngine

# Create engine
engine = InferenceEngine("mlx-community/Qwen3-8B-4bit")

# Generate response
result = engine.generate(messages, params)
```

## API Endpoints

- `POST /v1/chat/completions` - Chat completions
- `POST /tokenize` - Tokenize text
- `POST /detokenize` - Detokenize tokens
- `GET /health` - Health check

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
