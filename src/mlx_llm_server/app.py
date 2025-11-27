"""FastAPI application for MLX LLM server."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
import time
import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from mlx_llm_server import __version__
from mlx_llm_server.config import settings
from mlx_llm_server.inference import MLXInferenceEngine
from mlx_llm_server.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Message,
    Choice,
    Usage,
    StreamResponse,
    StreamChoice,
    DeltaMessage,
    ModelObject,
    ModelsResponse,
)


# Global inference engine
engine: MLXInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global engine
    engine = MLXInferenceEngine(settings.model_name)
    yield
    engine = None


app = FastAPI(
    title="MLX LLM Server",
    description="MLX-based LLM inference server for Apple Silicon",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=settings.model_name,
        version=__version__,
    )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """OpenAI-compatible models list endpoint."""
    return ModelsResponse(
        data=[
            ModelObject(
                id=settings.model_name,
                created=int(time.time()),
                owned_by="mlx-community",
            )
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        return {"error": "Model not loaded"}

    # Convert Pydantic models to dicts for the template
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = engine.apply_chat_template(messages)

    # Generate unique ID and timestamp
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            stream_response(prompt, request, completion_id, created),
            media_type="text/event-stream",
        )
    else:
        # Count prompt tokens (rough estimate)
        prompt_tokens = len(engine.tokenizer.encode(prompt))

        response_text = engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens or settings.max_tokens,
            temperature=request.temperature or settings.temperature,
            top_p=request.top_p or settings.top_p,
            stream=False,
        )

        # Count completion tokens
        completion_tokens = len(engine.tokenizer.encode(response_text))

        return ChatResponse(
            id=completion_id,
            created=created,
            model=settings.model_name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


async def stream_response(
    prompt: str, request: ChatRequest, completion_id: str, created: int
) -> AsyncIterator[str]:
    """Stream response chunks in OpenAI SSE format."""
    if engine is None:
        yield 'data: {"error": "Model not loaded"}\n\n'
        return

    # Send first chunk with role
    first_chunk = StreamResponse(
        id=completion_id,
        created=created,
        model=settings.model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # Stream content chunks
    for chunk in engine.generate(
        prompt=prompt,
        max_tokens=request.max_tokens or settings.max_tokens,
        temperature=request.temperature or settings.temperature,
        top_p=request.top_p or settings.top_p,
        stream=True,
    ):
        chunk_response = StreamResponse(
            id=completion_id,
            created=created,
            model=settings.model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=chunk),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk_response.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = StreamResponse(
        id=completion_id,
        created=created,
        model=settings.model_name,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def run_server(host: str = None, port: int = None, model_name: str = None):
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        model_name: Model name to use
    """
    if host:
        settings.host = host
    if port:
        settings.port = port
    if model_name:
        settings.model_name = model_name

    uvicorn.run(
        "mlx_llm_server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
