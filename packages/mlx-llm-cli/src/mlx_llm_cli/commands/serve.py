"""서버 실행 커맨드 (다중 모델 지원)"""
import typer
from rich.console import Console
from typing import Optional
from pathlib import Path

app = typer.Typer(help="서버 관련 명령어")
console = Console()

DEFAULT_PORT = 9044
DEFAULT_HOST = "0.0.0.0"


@app.command("start")
def start(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="YAML 설정 파일 경로",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port", "-p",
        help="서버 포트 (설정 파일보다 우선)",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host", "-h",
        help="서버 호스트 (설정 파일보다 우선)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="개발 모드 (자동 리로드)",
    ),
) -> None:
    """서버 시작 (다중 모델 지원)"""
    import uvicorn
    from mlx_llm_core import load_config, merge_config, AppConfig, Success, Failure

    # 설정 로드
    if config_path:
        config_result = load_config(config_path)
        if isinstance(config_result, Failure):
            console.print(f"[bold red]Error loading config: {config_result.error}[/bold red]")
            raise typer.Exit(1)
        config = config_result.value
        console.print(f"[dim]Config loaded from: {config_path}[/dim]")
    else:
        # 기본 경로에서 자동 로드 시도
        config_result = load_config()
        if isinstance(config_result, Success):
            config = config_result.value
            console.print("[dim]Config loaded from default location[/dim]")
        else:
            config = AppConfig()
            console.print("[dim]Using default config[/dim]")

    # CLI 옵션으로 오버라이드
    overrides = {}
    if port is not None:
        overrides["server"] = {"port": port}
    if host is not None:
        if "server" not in overrides:
            overrides["server"] = {}
        overrides["server"]["host"] = host

    if overrides:
        config = merge_config(config, overrides)

    # 서버 정보 출력
    console.print(f"[bold blue]Starting MLX LLM Server (Multi-Model)...[/bold blue]")
    console.print(f"  Host:    [cyan]{config.server.host}:{config.server.port}[/cyan]")
    console.print(f"  Default: [green]{config.models.default}[/green]")
    console.print(f"  Models:  [yellow]{', '.join(config.models.list_aliases())}[/yellow]")

    from mlx_llm_api import create_app

    fastapi_app = create_app(config)

    uvicorn.run(
        fastapi_app,
        host=config.server.host,
        port=config.server.port,
        reload=reload,
    )


@app.command("config")
def show_config(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="YAML 설정 파일 경로",
    ),
) -> None:
    """현재 설정 출력"""
    from mlx_llm_core import load_config, AppConfig, Success, Failure
    from rich.table import Table
    import yaml

    if config_path:
        config_result = load_config(config_path)
        if isinstance(config_result, Failure):
            console.print(f"[bold red]Error loading config: {config_result.error}[/bold red]")
            raise typer.Exit(1)
        config = config_result.value
    else:
        config_result = load_config()
        config = config_result.value if isinstance(config_result, Success) else AppConfig()

    # 설정 출력
    console.print("[bold blue]Current Configuration[/bold blue]\n")

    # 서버 설정
    console.print("[bold]Server:[/bold]")
    console.print(f"  host: {config.server.host}")
    console.print(f"  port: {config.server.port}")

    # 모델 설정
    console.print("\n[bold]Models:[/bold]")
    console.print(f"  default: {config.models.default}")
    console.print(f"  max_loaded: {config.models.max_loaded}")

    table = Table(title="Available Models")
    table.add_column("Alias", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Context", style="yellow")
    table.add_column("Description")

    for alias, defn in config.models.available.items():
        table.add_row(
            alias,
            defn.path,
            str(defn.context_length),
            defn.description,
        )

    console.print(table)

    # 추론 설정
    console.print("\n[bold]Inference:[/bold]")
    console.print(f"  max_tokens: {config.inference.max_tokens}")
    console.print(f"  temperature: {config.inference.temperature}")
    console.print(f"  top_p: {config.inference.top_p}")


@app.callback(invoke_without_command=True)
def serve_callback(ctx: typer.Context) -> None:
    """서버 명령어 (기본: start)"""
    if ctx.invoked_subcommand is None:
        start()
