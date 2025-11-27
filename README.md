# MLX LLM Server

MLX-based LLM inference server for Apple Silicon, optimized for Qwen models.

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.10-3.12** (MLX does not support Python 3.13+)
- Poetry for dependency management

## Installation

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Usage

### Start the server

Default configuration (Qwen3-8B-4bit on port 9044):
```bash
poetry run mlx-llm-server
```

Or with custom model:
```bash
poetry run mlx-llm-server --model-name mlx-community/Qwen2.5-7B-Instruct-4bit --port 9044
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:9044/health
```

#### Chat Completion
```bash
curl -X POST http://localhost:9044/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 2048,
    "temperature": 0.7
  }'
```

#### Streaming Chat
```bash
curl -X POST http://localhost:9044/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:9044/docs
- ReDoc: http://localhost:9044/redoc

## Available Models

The server works with any MLX-quantized model from HuggingFace. Recommended models:

- `mlx-community/Qwen3-8B-4bit` (default)
- `mlx-community/Qwen2.5-7B-Instruct-4bit`
- `mlx-community/Qwen2.5-14B-Instruct-4bit`

## CLI Options

```
mlx-llm-server --help

Options:
  --host TEXT         Host to bind to [default: 0.0.0.0]
  --port INTEGER      Port to bind to [default: 9044]
  --model-name TEXT   HuggingFace model name or path
  --version          Show version
  --help             Show help message
```

## Development

### Run tests
```bash
poetry run pytest
```

### Format code
```bash
poetry run black src tests
poetry run ruff check src tests
```

## Project Structure

```
mlx_llm/
├── src/mlx_llm_server/
│   ├── __init__.py         # Package version
│   ├── app.py              # FastAPI application
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration settings
│   ├── inference.py        # MLX inference engine
│   └── models.py           # Pydantic models
├── tests/                  # Test files
├── pyproject.toml          # Poetry configuration
└── README.md               # This file
```

## License

MIT
