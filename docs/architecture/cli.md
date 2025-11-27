# CLI 패키지 (mlx-llm-cli)

Click 기반 CLI 도구입니다. 사용자와의 I/O 경계입니다.

## 모듈 구조

```
mlx_llm_cli/
├── __init__.py
├── main.py           # CLI 진입점
├── commands/         # 명령어 그룹
│   ├── __init__.py
│   ├── serve.py      # mlx-llm serve
│   ├── chat.py       # mlx-llm chat
│   ├── rag.py        # mlx-llm rag
│   └── config.py     # mlx-llm config
└── formatters.py     # Rich 출력
```

## main.py - CLI 진입점

```python
"""CLI 진입점"""
import click
from pathlib import Path

from mlx_llm_core import AppConfig
from mlx_llm_cli.commands import serve, chat, rag, config as config_cmd


@click.group()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, path_type=Path),
    help='설정 파일 경로'
)
@click.version_option(version='0.1.0')
@click.pass_context
def cli(ctx: click.Context, config: Path | None):
    """
    MLX LLM - Apple Silicon LLM Platform

    OpenAI 호환 LLM 서버 및 CLI 도구
    """
    ctx.ensure_object(dict)

    # 설정 로드
    if config:
        import yaml
        with open(config) as f:
            config_data = yaml.safe_load(f)
        ctx.obj['config'] = AppConfig(**config_data)
    else:
        ctx.obj['config'] = AppConfig()


# 명령어 등록
cli.add_command(serve.serve)
cli.add_command(chat.chat)
cli.add_command(rag.rag)
cli.add_command(config_cmd.config)


if __name__ == '__main__':
    cli()
```

## commands/serve.py - 서버 명령

```python
"""서버 명령"""
import click
from rich.console import Console

console = Console()


@click.command()
@click.option('--host', '-h', default=None, help='호스트 (기본: 127.0.0.1)')
@click.option('--port', '-p', default=None, type=int, help='포트 (기본: 9044)')
@click.option('--model', '-m', default=None, help='모델 이름')
@click.option('--reload', is_flag=True, help='자동 리로드 (개발용)')
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None, model: str | None, reload: bool):
    """
    HTTP API 서버 실행

    예시:
        mlx-llm serve
        mlx-llm serve --port 8080
        mlx-llm serve --model mlx-community/Qwen3-8B-4bit
    """
    import uvicorn
    from mlx_llm_core import AppConfig
    from mlx_llm_api import create_app

    config: AppConfig = ctx.obj['config']

    # CLI 옵션으로 오버라이드
    final_host = host or config.server.host
    final_port = port or config.server.port

    if model:
        # 새 config 생성 (불변)
        config = AppConfig(
            server=config.server,
            model=config.model.model_copy(update={'name': model}),
            inference=config.inference,
        )

    console.print(f"[bold green]Starting MLX LLM Server[/bold green]")
    console.print(f"  Host: {final_host}")
    console.print(f"  Port: {final_port}")
    console.print(f"  Model: {config.model.name}")

    app = create_app(config)

    uvicorn.run(
        app,
        host=final_host,
        port=final_port,
        reload=reload,
    )
```

## commands/chat.py - 채팅 명령

```python
"""채팅 명령"""
import click
from rich.console import Console
from rich.markdown import Markdown

from mlx_llm_core import (
    Message, GenerationParams, NonEmptyList,
    Temperature, TopP, MaxTokens,
    Success, Failure,
)

console = Console()


@click.command()
@click.argument('prompt', required=False)
@click.option('--model', '-m', default=None, help='모델 이름')
@click.option('--temperature', '-t', default=0.7, type=float, help='Temperature')
@click.option('--max-tokens', default=2048, type=int, help='최대 토큰')
@click.option('--system', '-s', default=None, help='시스템 프롬프트')
@click.option('--stream/--no-stream', default=True, help='스트리밍 출력')
@click.pass_context
def chat(
    ctx: click.Context,
    prompt: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int,
    system: str | None,
    stream: bool
):
    """
    대화형 채팅

    예시:
        mlx-llm chat "Hello"
        mlx-llm chat --system "You are a poet"
        mlx-llm chat  # 대화형 모드
    """
    from mlx_llm_core import AppConfig
    from mlx_llm_inference import InferenceEngine

    config: AppConfig = ctx.obj['config']
    model_name = model or config.model.name

    console.print(f"[dim]Loading model: {model_name}[/dim]")
    engine = InferenceEngine(model_name)

    params = GenerationParams(
        max_tokens=MaxTokens(max_tokens),
        temperature=Temperature(temperature),
        top_p=TopP.default(),
    )

    if prompt:
        # 단일 프롬프트 모드
        _generate_response(engine, prompt, system, params, stream)
    else:
        # 대화형 모드
        _interactive_mode(engine, system, params, stream)


def _generate_response(
    engine: 'InferenceEngine',
    prompt: str,
    system: str | None,
    params: GenerationParams,
    stream: bool
):
    """응답 생성"""
    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=prompt))

    messages_result = NonEmptyList.of(messages)
    match messages_result:
        case Failure(err):
            console.print(f"[red]Error: {err}[/red]")
            return
        case Success(non_empty):
            pass

    if stream:
        console.print("[bold]Assistant:[/bold]")
        for chunk in engine.stream(non_empty, params):
            console.print(chunk, end="")
        console.print()
    else:
        result = engine.generate(non_empty, params)
        match result:
            case Success(response):
                console.print("[bold]Assistant:[/bold]")
                console.print(Markdown(response.content))
            case Failure(err):
                console.print(f"[red]Error: {err}[/red]")


def _interactive_mode(
    engine: 'InferenceEngine',
    system: str | None,
    params: GenerationParams,
    stream: bool
):
    """대화형 모드"""
    console.print("[bold green]Interactive Mode[/bold green] (Ctrl+C to exit)")

    history: list[Message] = []
    if system:
        history.append(Message(role="system", content=system))

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            if not user_input.strip():
                continue

            history.append(Message(role="user", content=user_input))

            messages_result = NonEmptyList.of(history)
            match messages_result:
                case Failure(err):
                    console.print(f"[red]Error: {err}[/red]")
                    continue
                case Success(non_empty):
                    pass

            console.print("[bold green]Assistant:[/bold green]")
            response_text = ""

            if stream:
                for chunk in engine.stream(non_empty, params):
                    console.print(chunk, end="")
                    response_text += chunk
                console.print()
            else:
                result = engine.generate(non_empty, params)
                match result:
                    case Success(response):
                        response_text = response.content
                        console.print(response_text)
                    case Failure(err):
                        console.print(f"[red]Error: {err}[/red]")
                        continue

            history.append(Message(role="assistant", content=response_text))

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
```

## Public API (__init__.py)

```python
"""MLX LLM CLI"""
from mlx_llm_cli.main import cli

__version__ = "0.1.0"

__all__ = ["cli"]
```

## 사용 예시

```bash
# 서버 실행
mlx-llm serve
mlx-llm serve --port 8080 --model mlx-community/Qwen3-8B-4bit

# 단일 채팅
mlx-llm chat "Hello, how are you?"
mlx-llm chat --system "You are a poet" "Write a haiku"

# 대화형 모드
mlx-llm chat

# 설정 파일 사용
mlx-llm --config ./config.yaml serve
```
