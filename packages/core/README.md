# zeta-mlx-core

Core types and pure functions for Zeta MLX platform.

## Installation

```bash
pip install zeta-mlx-core
```

## Features

- **Pure Domain Types**: Message, GenerationParams, ToolDefinition
- **Result Monad**: Railway-oriented programming with `Result[T, E]`
- **Validation**: Pure validation functions
- **Configuration**: YAML-based config management

## Usage

```python
from zeta_mlx.core import (
    Message, GenerationParams, Result, Success, Failure,
    Temperature, TopP, MaxTokens,
)

# Create a message
msg = Message(role="user", content="Hello!")

# Create generation parameters
params = GenerationParams(
    max_tokens=MaxTokens(1024),
    temperature=Temperature(0.7),
    top_p=TopP(0.9),
)

# Use Result for error handling
result: Result[str, str] = Success("Hello!")
```

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
