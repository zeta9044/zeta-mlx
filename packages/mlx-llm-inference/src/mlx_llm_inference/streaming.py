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
    """
    prev_text = ""
    sampler = make_sampler(temp=temperature, top_p=top_p)

    for response in stream_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        # 새로운 텍스트만 추출
        current_text = response.text
        new_text = current_text[len(prev_text):]

        if new_text:
            yield new_text
            prev_text = current_text


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
