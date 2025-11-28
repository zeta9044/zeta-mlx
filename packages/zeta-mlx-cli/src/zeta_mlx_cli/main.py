"""Zeta MLX CLI 메인 엔트리"""
import typer
from rich.console import Console
from zeta_mlx_cli.commands import serve, chat, models, embedding

console = Console()

app = typer.Typer(
    name="zeta-mlx",
    help="Zeta MLX - OpenAI-compatible LLM/Embedding inference on Apple Silicon",
    add_completion=False,
)

# 서브커맨드 등록
app.add_typer(serve.app, name="serve", help="LLM 서버 (포트 9044)")
app.add_typer(embedding.app, name="embedding", help="임베딩 서버 (포트 9045)")
app.add_typer(chat.app, name="chat")
app.add_typer(models.app, name="models")


@app.callback()
def main_callback() -> None:
    """Zeta MLX CLI"""
    pass


@app.command()
def version() -> None:
    """버전 정보 출력"""
    from zeta_mlx_cli import __version__
    console.print(f"[bold blue]zeta-mlx[/bold blue] version [green]{__version__}[/green]")


def cli() -> None:
    """CLI 진입점"""
    app()


if __name__ == "__main__":
    cli()
