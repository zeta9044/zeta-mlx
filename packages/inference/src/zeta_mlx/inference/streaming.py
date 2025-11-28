"""스트리밍 Generator"""
from typing import Iterator, Any
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler


def mlx_stream_generator(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterator[str]:
    """
    MLX 스트리밍 Generator

    mlx_lm.stream_generate를 사용하여 텍스트를 스트리밍합니다.
    각 response.text는 이미 개별 청크이므로 직접 yield합니다.
    """
    sampler = make_sampler(temp=temperature, top_p=top_p)

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        if response.text:
            yield response.text


def chunk_stream(
    stream: Iterator[str],
    min_chunk_size: int = 1,
) -> Iterator[str]:
    """
    스트림 청크 조절

    너무 작은 청크를 모아서 전송합니다.
    """
    buffer = ""

    for chunk in stream:
        buffer += chunk

        if len(buffer) >= min_chunk_size:
            yield buffer
            buffer = ""

    if buffer:
        yield buffer
