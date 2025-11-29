# zeta-mlx-langchain

LangChain integration for Zeta MLX platform.

## Installation

```bash
pip install zeta-mlx-langchain
```

## Features

- **ChatModel**: LangChain BaseChatModel implementation
- **Embeddings**: LangChain Embeddings implementation
- **Streaming**: Async streaming support

## Usage

```python
from zeta_mlx.langchain import ChatZetaMLX

# Create chat model
llm = ChatZetaMLX(model="qwen3-8b")

# Use with LangChain
from langchain_core.messages import HumanMessage

response = llm.invoke([HumanMessage(content="Hello!")])

# Streaming
for chunk in llm.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="")
```

## Links

- [GitHub](https://github.com/zeta9044/zeta-mlx)
- [Documentation](https://github.com/zeta9044/zeta-mlx#readme)
