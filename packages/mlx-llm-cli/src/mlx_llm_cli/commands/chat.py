"""대화형 채팅 커맨드"""
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from typing import Optional

app = typer.Typer(help="대화형 채팅")
console = Console()

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"


@app.command("interactive")
def interactive(
    model_name: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="HuggingFace 모델 이름",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="생성 온도 (0.0-2.0)",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens",
        help="최대 생성 토큰 수",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system", "-s",
        help="시스템 프롬프트",
    ),
) -> None:
    """대화형 채팅 시작"""
    from mlx_llm_core import Message, GenerationParams, ChatRequest, Temperature, TopP, MaxTokens
    from mlx_llm_inference import load_model_safe, InferenceEngine, Failure

    console.print(f"[bold blue]Loading model: {model_name}...[/bold blue]")

    # 모델 로드
    bundle_result = load_model_safe(model_name)
    if isinstance(bundle_result, Failure):
        console.print(f"[bold red]Error loading model: {bundle_result.error}[/bold red]")
        raise typer.Exit(1)

    engine = InferenceEngine(bundle_result.value)
    console.print(f"[bold green]Model loaded successfully![/bold green]")
    console.print("[dim]Type 'exit' or 'quit' to end the conversation.[/dim]\n")

    # 대화 이력
    messages: list[Message] = []

    # 시스템 프롬프트 추가
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))

    # Temperature, MaxTokens 생성
    temp_result = Temperature.create(temperature)
    max_tokens_result = MaxTokens.create(max_tokens)

    if isinstance(temp_result, Failure) or isinstance(max_tokens_result, Failure):
        console.print("[bold red]Invalid parameters[/bold red]")
        raise typer.Exit(1)

    params = GenerationParams(
        temperature=temp_result.value,
        max_tokens=max_tokens_result.value,
    )

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            # 사용자 메시지 추가
            messages.append(Message(role="user", content=user_input))

            # 요청 생성
            request = ChatRequest(
                model=model_name,
                messages=messages,
                params=params,
                stream=True,
            )

            # 스트리밍 생성
            console.print("[bold green]Assistant[/bold green]: ", end="")
            full_response = ""

            for chunk in engine.stream(request):
                console.print(chunk, end="")
                full_response += chunk

            console.print()  # 줄바꿈

            # 어시스턴트 응답 이력에 추가
            messages.append(Message(role="assistant", content=full_response))

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Goodbye![/dim]")
            break


@app.command("once")
def once(
    prompt: str = typer.Argument(..., help="프롬프트"),
    model_name: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="HuggingFace 모델 이름",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="생성 온도",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens",
        help="최대 생성 토큰 수",
    ),
) -> None:
    """단일 프롬프트로 응답 생성"""
    from mlx_llm_core import Message, GenerationParams, ChatRequest, Temperature, MaxTokens, Failure
    from mlx_llm_inference import load_model_safe, InferenceEngine

    console.print(f"[dim]Loading model: {model_name}...[/dim]")

    bundle_result = load_model_safe(model_name)
    if isinstance(bundle_result, Failure):
        console.print(f"[bold red]Error: {bundle_result.error}[/bold red]")
        raise typer.Exit(1)

    engine = InferenceEngine(bundle_result.value)

    temp_result = Temperature.create(temperature)
    max_tokens_result = MaxTokens.create(max_tokens)

    if isinstance(temp_result, Failure) or isinstance(max_tokens_result, Failure):
        console.print("[bold red]Invalid parameters[/bold red]")
        raise typer.Exit(1)

    params = GenerationParams(
        temperature=temp_result.value,
        max_tokens=max_tokens_result.value,
    )

    messages = [Message(role="user", content=prompt)]
    request = ChatRequest(
        model=model_name,
        messages=messages,
        params=params,
        stream=False,
    )

    result = engine.generate(request)
    if isinstance(result, Failure):
        console.print(f"[bold red]Error: {result.error}[/bold red]")
        raise typer.Exit(1)

    console.print(Markdown(result.value.text))


@app.callback(invoke_without_command=True)
def chat_callback(ctx: typer.Context) -> None:
    """채팅 명령어 (기본: interactive)"""
    if ctx.invoked_subcommand is None:
        interactive()
