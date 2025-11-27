"""CLI entry point for MLX LLM server."""

import click
from mlx_llm_server.app import run_server
from mlx_llm_server import __version__


@click.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to",
    show_default=True,
)
@click.option(
    "--port",
    default=9044,
    help="Port to bind to",
    show_default=True,
    type=int,
)
@click.option(
    "--model-name",
    default="mlx-community/Qwen3-8B-4bit",
    help="HuggingFace model name or path",
    show_default=True,
)
@click.version_option(version=__version__, prog_name="mlx-llm-server")
def main(host: str, port: int, model_name: str):
    """MLX-based LLM inference server for Apple Silicon.

    Example:
        mlx-llm-server --port 9044 --model-name mlx-community/Qwen3-8B-4bit
    """
    click.echo(f"Starting MLX LLM Server v{__version__}")
    click.echo(f"Model: {model_name}")
    click.echo(f"Server: http://{host}:{port}")
    click.echo(f"API Docs: http://{host}:{port}/docs")

    run_server(host=host, port=port, model_name=model_name)


if __name__ == "__main__":
    main()
