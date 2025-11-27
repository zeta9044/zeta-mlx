"""MLX LLM CLI 메인 엔트리"""
import typer
from rich.console import Console
from mlx_llm_cli.commands import serve, chat, models

console = Console()

app = typer.Typer(
    name="mlx-llm",
    help="MLX LLM - OpenAI-compatible LLM inference on Apple Silicon",
    add_completion=False,
)

# 서브커맨드 등록
app.add_typer(serve.app, name="serve")
app.add_typer(chat.app, name="chat")
app.add_typer(models.app, name="models")


@app.callback()
def main_callback() -> None:
    """MLX LLM CLI"""
    pass


@app.command()
def version() -> None:
    """버전 정보 출력"""
    from mlx_llm_cli import __version__
    console.print(f"[bold blue]mlx-llm[/bold blue] version [green]{__version__}[/green]")


def cli() -> None:
    """CLI 진입점"""
    app()


if __name__ == "__main__":
    cli()
