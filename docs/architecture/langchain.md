# LangChain 패키지 (packages/langchain)

LangChain 생태계와의 통합 어댑터입니다.

## 모듈 구조

```
zeta_mlx/langchain/
├── __init__.py
├── chat_model.py     # BaseChatModel 구현
├── embeddings.py     # Embeddings 구현
└── tools.py          # Tool 정의
```

## chat_model.py - LangChain ChatModel

```python
"""LangChain BaseChatModel 구현"""
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, AIMessage, AIMessageChunk,
    HumanMessage, SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

from zeta_mlx.core import (
    Message, GenerationParams, NonEmptyList,
    Temperature, TopP, MaxTokens,
    Success, Failure,
)
from zeta_mlx.inference import InferenceEngine


class MLXChatModel(BaseChatModel):
    """
    MLX 기반 LangChain ChatModel

    사용 예:
        llm = MLXChatModel(model="mlx-community/Qwen3-8B-4bit")
        response = llm.invoke([HumanMessage(content="Hello")])
    """

    model: str = Field(default="mlx-community/Qwen3-8B-4bit")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    _engine: Optional[InferenceEngine] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "mlx-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

    def _get_engine(self) -> InferenceEngine:
        """지연 로딩으로 엔진 초기화"""
        if self._engine is None:
            self._engine = InferenceEngine(self.model)
        return self._engine

    def _convert_messages(self, messages: List[BaseMessage]) -> NonEmptyList[Message]:
        """LangChain 메시지 → 내부 타입 변환"""
        converted = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"

            converted.append(Message(role=role, content=msg.content))

        result = NonEmptyList.of(converted)
        match result:
            case Success(non_empty):
                return non_empty
            case Failure(err):
                raise ValueError(f"Invalid messages: {err}")

    def _get_params(self) -> GenerationParams:
        """파라미터 생성"""
        return GenerationParams(
            max_tokens=MaxTokens(self.max_tokens),
            temperature=Temperature(self.temperature),
            top_p=TopP(self.top_p),
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 생성"""
        engine = self._get_engine()
        internal_messages = self._convert_messages(messages)
        params = self._get_params()

        # 스트리밍으로 생성 (콜백 지원)
        response_text = ""
        for chunk in engine.stream(internal_messages, params):
            response_text += chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk)

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """스트리밍 생성"""
        engine = self._get_engine()
        internal_messages = self._convert_messages(messages)
        params = self._get_params()

        for chunk in engine.stream(internal_messages, params):
            if run_manager:
                run_manager.on_llm_new_token(chunk)

            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk)
            )
```

## embeddings.py - LangChain Embeddings

```python
"""LangChain Embeddings 구현"""
from typing import List

from langchain_core.embeddings import Embeddings
from pydantic import Field

from zeta_mlx.rag import create_sentence_transformer_embedder, EmbedFn


class MLXEmbeddings(Embeddings):
    """
    MLX/SentenceTransformer 기반 LangChain Embeddings

    사용 예:
        embeddings = MLXEmbeddings()
        vectors = embeddings.embed_documents(["Hello", "World"])
    """

    model: str = Field(default="all-MiniLM-L6-v2")
    _embed_fn: EmbedFn | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = create_sentence_transformer_embedder(self.model)
        return self._embed_fn

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩"""
        embed_fn = self._get_embed_fn()
        return embed_fn(texts)

    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩"""
        embed_fn = self._get_embed_fn()
        return embed_fn([text])[0]
```

## tools.py - LangChain Tools

```python
"""LangChain Tool 정의"""
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from zeta_mlx.inference import InferenceEngine


class TokenCountInput(BaseModel):
    """토큰 카운트 입력"""
    text: str = Field(description="토큰 수를 계산할 텍스트")


class TokenCountTool(BaseTool):
    """
    토큰 카운트 도구

    사용 예:
        tool = TokenCountTool()
        count = tool.invoke({"text": "Hello world"})
    """

    name: str = "token_count"
    description: str = "텍스트의 토큰 수를 계산합니다"
    args_schema: Type[BaseModel] = TokenCountInput

    model: str = Field(default="mlx-community/Qwen3-8B-4bit")
    _engine: Optional[InferenceEngine] = None

    def _get_engine(self) -> InferenceEngine:
        if self._engine is None:
            self._engine = InferenceEngine(self.model)
        return self._engine

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        engine = self._get_engine()
        count = engine.count_tokens(text)
        return f"토큰 수: {count}"
```

## Public API (__init__.py)

```python
"""Zeta MLX LangChain - LangChain Integration"""
from zeta_mlx.langchain.chat_model import MLXChatModel
from zeta_mlx.langchain.embeddings import MLXEmbeddings
from zeta_mlx.langchain.tools import TokenCountTool

__version__ = "0.1.0"

__all__ = [
    "MLXChatModel",
    "MLXEmbeddings",
    "TokenCountTool",
]
```

## 사용 예시

### 기본 채팅

```python
from zeta_mlx.langchain import MLXChatModel
from langchain_core.messages import HumanMessage, SystemMessage

llm = MLXChatModel(
    model="mlx-community/Qwen3-8B-4bit",
    temperature=0.7,
)

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
])
print(response.content)
```

### 스트리밍

```python
for chunk in llm.stream([HumanMessage(content="Tell me a story")]):
    print(chunk.content, end="", flush=True)
```

### Chain 구성

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "role": "Python expert",
    "question": "Explain decorators",
})
```

### RAG Chain

```python
from zeta_mlx.langchain import MLXChatModel, MLXEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

embeddings = MLXEmbeddings()
llm = MLXChatModel()

vectorstore = FAISS.load_local("./index", embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Context: {context}

Question: {question}

Answer:
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

result = rag_chain.invoke("What is the main feature?")
```

### Tool Calling Agent

```python
from zeta_mlx.langchain import MLXChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 조회합니다."""
    return f"{city}: 맑음, 22°C"

@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    return str(eval(expression))

llm = MLXChatModel(temperature=0)
agent = create_react_agent(llm, [get_weather, calculate])

result = agent.invoke({
    "messages": [HumanMessage(content="서울 날씨와 123*456 계산해줘")]
})
```
