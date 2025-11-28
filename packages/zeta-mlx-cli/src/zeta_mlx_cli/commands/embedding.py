"""임베딩 서버 실행 커맨드"""
import typer
import os
import sys
import signal
import subprocess
from rich.console import Console
from typing import Optional
from pathlib import Path

app = typer.Typer(help="임베딩 서버 관련 명령어")
console = Console()

DEFAULT_PORT = 9045
DEFAULT_HOST = "0.0.0.0"
PID_FILE = Path.home() / ".zeta-mlx" / "embedding.pid"
LOG_FILE = Path.home() / ".zeta-mlx" / "embedding.log"


def _ensure_dir() -> None:
    """PID/로그 디렉토리 생성"""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_pid() -> int | None:
    """PID 파일 읽기"""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            return None
    return None


def _write_pid(pid: int) -> None:
    """PID 파일 쓰기"""
    _ensure_dir()
    PID_FILE.write_text(str(pid))


def _remove_pid() -> None:
    """PID 파일 삭제"""
    if PID_FILE.exists():
        PID_FILE.unlink()


def _is_running(pid: int) -> bool:
    """프로세스 실행 중인지 확인"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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
        "--host",
        help="서버 호스트 (설정 파일보다 우선)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="임베딩 모델 별칭 (설정 파일보다 우선)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="개발 모드 (자동 리로드)",
    ),
    daemon: bool = typer.Option(
        False,
        "--daemon", "-d",
        help="백그라운드 데몬으로 실행",
    ),
) -> None:
    """임베딩 서버 시작"""
    # 이미 실행 중인지 확인
    existing_pid = _read_pid()
    if existing_pid and _is_running(existing_pid):
        console.print(f"[bold yellow]Embedding server already running (PID: {existing_pid})[/bold yellow]")
        console.print("Use 'zeta-mlx embedding stop' to stop it first.")
        raise typer.Exit(1)

    if daemon:
        _start_daemon(config_path, port, host, model)
    else:
        _start_foreground(config_path, port, host, model, reload)


def _start_daemon(
    config_path: Optional[Path],
    port: Optional[int],
    host: Optional[str],
    model: Optional[str],
) -> None:
    """백그라운드 데몬으로 시작"""
    _ensure_dir()

    # 커맨드 구성
    cmd = [sys.executable, "-m", "zeta_mlx_cli.commands.embedding_worker"]
    if config_path:
        cmd.extend(["--config", str(config_path)])
    if port:
        cmd.extend(["--port", str(port)])
    if host:
        cmd.extend(["--host", host])
    if model:
        cmd.extend(["--model", model])

    # 로그 파일 열기
    log_file = open(LOG_FILE, "a")

    # 백그라운드 프로세스 시작
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    _write_pid(process.pid)

    console.print(f"[bold green]Embedding server started in background[/bold green]")
    console.print(f"  PID:  [cyan]{process.pid}[/cyan]")
    console.print(f"  Log:  [dim]{LOG_FILE}[/dim]")
    console.print(f"\nUse 'zeta-mlx embedding status' to check status")
    console.print(f"Use 'zeta-mlx embedding stop' to stop the server")


def _start_foreground(
    config_path: Optional[Path],
    port: Optional[int],
    host: Optional[str],
    model: Optional[str],
    reload: bool,
) -> None:
    """포그라운드에서 시작"""
    import uvicorn
    from zeta_mlx_core import load_config, merge_config, AppConfig, Success, Failure

    # 설정 로드
    if config_path:
        config_result = load_config(config_path)
        if isinstance(config_result, Failure):
            console.print(f"[bold red]Error loading config: {config_result.error}[/bold red]")
            raise typer.Exit(1)
        config = config_result.value
        console.print(f"[dim]Config loaded from: {config_path}[/dim]")
    else:
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
        overrides["embedding_server"] = {"port": port}
    if host is not None:
        if "embedding_server" not in overrides:
            overrides["embedding_server"] = {}
        overrides["embedding_server"]["host"] = host
    if model is not None:
        overrides["embedding_models"] = {"default": model}

    if overrides:
        config = merge_config(config, overrides)

    # 모델 정보 가져오기
    model_alias = config.embedding_models.default
    model_def = config.embedding_models.get_model(model_alias)

    if model_def is None:
        console.print(f"[bold red]Model '{model_alias}' not found in config[/bold red]")
        console.print(f"Available models: {', '.join(config.embedding_models.list_aliases())}")
        raise typer.Exit(1)

    # 서버 정보 출력
    console.print(f"[bold blue]Starting Zeta MLX Embedding Server...[/bold blue]")
    console.print(f"  Host:    [cyan]{config.embedding_server.host}:{config.embedding_server.port}[/cyan]")
    console.print(f"  Model:   [green]{model_alias}[/green] ({model_def.path})")
    console.print(f"  Models:  [yellow]{', '.join(config.embedding_models.list_aliases())}[/yellow]")

    from zeta_mlx_embedding.api import create_app

    fastapi_app = create_app(config=config)

    # PID 저장 (foreground에서도)
    _write_pid(os.getpid())

    try:
        uvicorn.run(
            fastapi_app,
            host=config.embedding_server.host,
            port=config.embedding_server.port,
            reload=reload,
        )
    finally:
        _remove_pid()


@app.command("stop")
def stop() -> None:
    """임베딩 서버 중지"""
    pid = _read_pid()

    if not pid:
        console.print("[yellow]No embedding server PID file found[/yellow]")
        raise typer.Exit(1)

    if not _is_running(pid):
        console.print(f"[yellow]Embedding server (PID: {pid}) is not running[/yellow]")
        _remove_pid()
        raise typer.Exit(1)

    console.print(f"Stopping embedding server (PID: {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
        console.print("[bold green]Embedding server stopped[/bold green]")
        _remove_pid()
    except OSError as e:
        console.print(f"[bold red]Failed to stop server: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("status")
def status() -> None:
    """임베딩 서버 상태 확인"""
    pid = _read_pid()

    if not pid:
        console.print("[dim]Embedding server is not running (no PID file)[/dim]")
        return

    if _is_running(pid):
        console.print(f"[bold green]Embedding server is running[/bold green]")
        console.print(f"  PID: [cyan]{pid}[/cyan]")
        console.print(f"  Log: [dim]{LOG_FILE}[/dim]")
    else:
        console.print(f"[yellow]Embedding server (PID: {pid}) is not running[/yellow]")
        _remove_pid()


@app.command("logs")
def logs(
    follow: bool = typer.Option(
        False,
        "--follow", "-f",
        help="실시간 로그 추적",
    ),
    lines: int = typer.Option(
        50,
        "--lines", "-n",
        help="출력할 줄 수",
    ),
) -> None:
    """임베딩 서버 로그 확인"""
    if not LOG_FILE.exists():
        console.print("[yellow]No log file found[/yellow]")
        raise typer.Exit(1)

    if follow:
        import time
        console.print(f"[dim]Following {LOG_FILE} (Ctrl+C to stop)...[/dim]\n")
        with open(LOG_FILE, "r") as f:
            f.seek(0, 2)
            try:
                while True:
                    line = f.readline()
                    if line:
                        console.print(line, end="")
                    else:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs[/dim]")
    else:
        with open(LOG_FILE, "r") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                console.print(line, end="")


@app.command("config")
def show_config(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="YAML 설정 파일 경로",
    ),
) -> None:
    """현재 임베딩 설정 출력"""
    from zeta_mlx_core import load_config, AppConfig, Success, Failure
    from rich.table import Table

    if config_path:
        config_result = load_config(config_path)
        if isinstance(config_result, Failure):
            console.print(f"[bold red]Error loading config: {config_result.error}[/bold red]")
            raise typer.Exit(1)
        config = config_result.value
    else:
        config_result = load_config()
        config = config_result.value if isinstance(config_result, Success) else AppConfig()

    console.print("[bold blue]Embedding Configuration[/bold blue]\n")

    console.print("[bold]Server:[/bold]")
    console.print(f"  host: {config.embedding_server.host}")
    console.print(f"  port: {config.embedding_server.port}")

    console.print("\n[bold]Models:[/bold]")
    console.print(f"  default: {config.embedding_models.default}")
    console.print(f"  max_loaded: {config.embedding_models.max_loaded}")
    console.print(f"  batch_size: {config.embedding_models.batch_size}")

    table = Table(title="Available Embedding Models")
    table.add_column("Alias", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Dimension")
    table.add_column("Description")

    for alias, defn in config.embedding_models.available.items():
        table.add_row(
            alias,
            defn.path,
            defn.provider,
            str(defn.dimension),
            defn.description,
        )

    console.print(table)


@app.callback(invoke_without_command=True)
def embedding_callback(ctx: typer.Context) -> None:
    """임베딩 서버 명령어 (기본: start)"""
    if ctx.invoked_subcommand is None:
        start()
