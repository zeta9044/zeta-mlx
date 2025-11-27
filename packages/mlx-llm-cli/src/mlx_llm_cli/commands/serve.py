"""서버 실행 커맨드"""
import typer
from rich.console import Console
from typing import Optional

app = typer.Typer(help="서버 관련 명령어")
console = Console()

DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_PORT = 9044
DEFAULT_HOST = "0.0.0.0"


@app.command("start")
def start(
    model_name: str = typer.Option(
        DEFAULT_MODEL,
        "--model", "-m",
        help="HuggingFace 모델 이름",
    ),
    port: int = typer.Option(
        DEFAULT_PORT,
        "--port", "-p",
        help="서버 포트",
    ),
    host: str = typer.Option(
        DEFAULT_HOST,
        "--host", "-h",
        help="서버 호스트",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="개발 모드 (자동 리로드)",
    ),
) -> None:
    """서버 시작"""
    import uvicorn
    from mlx_llm_core import ServerConfig
    from mlx_llm_api import create_app

    console.print(f"[bold blue]Starting MLX LLM Server...[/bold blue]")
    console.print(f"  Model: [green]{model_name}[/green]")
    console.print(f"  Host:  [cyan]{host}:{port}[/cyan]")

    config = ServerConfig(
        model_name=model_name,
        host=host,
        port=port,
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


@app.callback(invoke_without_command=True)
def serve_callback(ctx: typer.Context) -> None:
    """서버 명령어 (기본: start)"""
    if ctx.invoked_subcommand is None:
        start()
