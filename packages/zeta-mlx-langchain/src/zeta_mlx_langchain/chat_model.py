"""LangChain Chat Model 통합"""
from typing import Any, Iterator, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from zeta_mlx_core import (
    Message, GenerationParams, ChatRequest,
    Temperature, TopP, MaxTokens,
    Success, Failure,
)
from zeta_mlx_inference import InferenceEngine, load_model_safe, ModelBundle


def _convert_message_to_domain(message: BaseMessage) -> Message:
    """LangChain 메시지를 도메인 메시지로 변환"""
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "user"

    content = message.content if isinstance(message.content, str) else str(message.content)
    return Message(role=role, content=content)


class ChatMLXLLM(BaseChatModel):
    """LangChain Chat Model for MLX LLM

    Example:
        from zeta_mlx_langchain import ChatMLXLLM

        llm = ChatMLXLLM(model_name="mlx-community/Qwen3-8B-4bit")
        response = llm.invoke("Hello, how are you?")
    """

    model_name: str = "mlx-community/Qwen3-8B-4bit"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048

    _engine: Optional[InferenceEngine] = None
    _bundle: Optional[ModelBundle] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "zeta-mlx"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def _ensure_engine(self) -> InferenceEngine:
        """엔진 초기화 (지연 로딩)"""
        if self._engine is None:
            result = load_model_safe(self.model_name)
            if isinstance(result, Failure):
                raise RuntimeError(f"Failed to load model: {result.error}")
            self._bundle = result.value
            self._engine = InferenceEngine(self._bundle)
        return self._engine

    def _create_params(self) -> GenerationParams:
        """생성 파라미터 생성"""
        temp_result = Temperature.create(self.temperature)
        top_p_result = TopP.create(self.top_p)
        max_tokens_result = MaxTokens.create(self.max_tokens)

        return GenerationParams(
            temperature=temp_result.value if isinstance(temp_result, Success) else None,
            top_p=top_p_result.value if isinstance(top_p_result, Success) else None,
            max_tokens=max_tokens_result.value if isinstance(max_tokens_result, Success) else None,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 생성"""
        engine = self._ensure_engine()

        # 메시지 변환
        domain_messages = [_convert_message_to_domain(m) for m in messages]

        # 요청 생성
        params = self._create_params()
        if stop:
            params = GenerationParams(
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                stop=stop,
            )

        request = ChatRequest(
            model=self.model_name,
            messages=domain_messages,
            params=params,
            stream=False,
        )

        # 생성
        result = engine.generate(request)

        if isinstance(result, Failure):
            raise RuntimeError(f"Generation failed: {result.error}")

        generation_result = result.value

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=generation_result.text),
                    generation_info={
                        "prompt_tokens": generation_result.prompt_tokens,
                        "completion_tokens": generation_result.completion_tokens,
                    },
                )
            ],
            llm_output={
                "model_name": self.model_name,
                "token_usage": {
                    "prompt_tokens": generation_result.prompt_tokens,
                    "completion_tokens": generation_result.completion_tokens,
                    "total_tokens": generation_result.prompt_tokens + generation_result.completion_tokens,
                },
            },
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """스트리밍 생성"""
        engine = self._ensure_engine()

        # 메시지 변환
        domain_messages = [_convert_message_to_domain(m) for m in messages]

        # 요청 생성
        params = self._create_params()
        if stop:
            params = GenerationParams(
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                stop=stop,
            )

        request = ChatRequest(
            model=self.model_name,
            messages=domain_messages,
            params=params,
            stream=True,
        )

        # 스트리밍 생성
        for chunk_text in engine.stream(request):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=chunk_text),
            )

            if run_manager:
                run_manager.on_llm_new_token(chunk_text)
