# zeta-mlx-cli

CLI for Zeta MLX - OpenAI-compatible LLM/Embedding inference on Apple Silicon.

## Installation

```bash
pip install zeta-mlx-cli
```

## Commands

```bash
# LLM Server (port 9044)
zeta-mlx llm start          # Start server
zeta-mlx llm start --daemon  # Start in background
zeta-mlx llm status         # Check status
zeta-mlx llm stop           # Stop server

# Embedding Server (port 9045)
zeta-mlx embedding start
zeta-mlx embedding status
zeta-mlx embedding stop

# Interactive chat
zeta-mlx chat

# Model management
zeta-mlx models list
```

## API Usage

```bash
# Chat completion
curl -X POST http://localhost:9044/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hello!"}]}'

# Tokenize
curl -X POST http://localhost:9044/tokenize \
  -d '{"prompt": "Hello, world!"}'
```

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
