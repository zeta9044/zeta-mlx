"""스트리밍 Generator"""
from typing import Iterator, Any
import mlx.core as mx
from mlx_lm.utils import generate_step


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

    UTF-8 멀티바이트 문자를 올바르게 처리합니다.
    토큰을 누적하고 완성된 문자만 yield합니다.
    """
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    token_buffer: list[int] = []
    prev_text = ""

    for token, _ in zip(
        generate_step(
            prompt_tokens,
            model,
            temp=temperature,
            top_p=top_p,
        ),
        range(max_tokens),
    ):
        # EOS 체크
        if token == tokenizer.eos_token_id:
            break

        # 토큰 ID 추출
        token_id = token[0] if isinstance(token, tuple) else token
        token_buffer.append(
            token_id.item() if hasattr(token_id, 'item') else token_id
        )

        # 누적 토큰 디코드
        try:
            current_text = tokenizer.decode(
                token_buffer,
                skip_special_tokens=True,
            )
        except Exception:
            continue

        # 새로운 텍스트만 추출
        new_text = current_text[len(prev_text):]

        # 불완전한 UTF-8 시퀀스 체크 (replacement character)
        if new_text and '\ufffd' not in new_text:
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
