"""모델 관리 커맨드"""
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="모델 관리")
console = Console()

# 추천 모델 목록
RECOMMENDED_MODELS = [
    ("mlx-community/Qwen3-8B-4bit", "Qwen3 8B (4-bit)", "8GB", "기본 추천"),
    ("mlx-community/Qwen3-4B-4bit", "Qwen3 4B (4-bit)", "4GB", "경량"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "Qwen2.5 7B (4-bit)", "7GB", "Instruction"),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "Llama 3.2 3B (4-bit)", "3GB", "경량"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "Mistral 7B (4-bit)", "7GB", "범용"),
]


@app.command("list")
def list_models() -> None:
    """추천 모델 목록 출력"""
    table = Table(title="추천 MLX 모델")
    table.add_column("모델", style="cyan")
    table.add_column("설명", style="green")
    table.add_column("크기", style="yellow")
    table.add_column("용도", style="magenta")

    for model_id, desc, size, usage in RECOMMENDED_MODELS:
        table.add_row(model_id, desc, size, usage)

    console.print(table)
    console.print("\n[dim]사용법: zeta-mlx serve start -m <model_id>[/dim]")


@app.command("info")
def model_info(
    model_name: str = typer.Argument(..., help="모델 이름"),
) -> None:
    """모델 정보 출력"""
    from zeta_mlx_inference import load_model_safe, Failure

    console.print(f"[dim]Loading model info: {model_name}...[/dim]")

    result = load_model_safe(model_name)
    if isinstance(result, Failure):
        console.print(f"[bold red]Error: {result.error}[/bold red]")
        raise typer.Exit(1)

    bundle = result.value

    table = Table(title=f"모델 정보: {model_name}")
    table.add_column("속성", style="cyan")
    table.add_column("값", style="green")

    # 토크나이저 정보
    if hasattr(bundle.tokenizer, "vocab_size"):
        table.add_row("Vocab Size", str(bundle.tokenizer.vocab_size))

    # 모델 config 정보 출력
    if hasattr(bundle.model, "args"):
        args = bundle.model.args
        if hasattr(args, "hidden_size"):
            table.add_row("Hidden Size", str(args.hidden_size))
        if hasattr(args, "num_hidden_layers"):
            table.add_row("Layers", str(args.num_hidden_layers))
        if hasattr(args, "num_attention_heads"):
            table.add_row("Attention Heads", str(args.num_attention_heads))

    console.print(table)


@app.command("download")
def download_model(
    model_name: str = typer.Argument(..., help="HuggingFace 모델 이름"),
) -> None:
    """모델 다운로드 (사전 캐싱)"""
    from zeta_mlx_inference import load_model_safe, Failure

    console.print(f"[bold blue]Downloading model: {model_name}...[/bold blue]")

    with console.status("[bold green]Downloading..."):
        result = load_model_safe(model_name)

    if isinstance(result, Failure):
        console.print(f"[bold red]Error: {result.error}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]✓ Model downloaded successfully![/bold green]")
