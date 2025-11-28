"""임베딩 서버 워커 (데몬용)"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, default=None)
    parser.add_argument("--port", "-p", type=int, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=None)
    args = parser.parse_args()

    from zeta_mlx_core import load_config, merge_config, AppConfig, Success, Failure
    from zeta_mlx_embedding.api import create_app

    # 설정 로드
    if args.config:
        config_result = load_config(args.config)
        if isinstance(config_result, Failure):
            print(f"Error loading config: {config_result.error}")
            return
        config = config_result.value
    else:
        config_result = load_config()
        config = config_result.value if isinstance(config_result, Success) else AppConfig()

    # CLI 오버라이드
    overrides = {}
    if args.port:
        overrides["embedding_server"] = {"port": args.port}
    if args.host:
        if "embedding_server" not in overrides:
            overrides["embedding_server"] = {}
        overrides["embedding_server"]["host"] = args.host
    if args.model:
        overrides["embedding_models"] = {"default": args.model}

    if overrides:
        config = merge_config(config, overrides)

    # 앱 생성 및 실행
    app = create_app(config=config)

    import uvicorn
    uvicorn.run(
        app,
        host=config.embedding_server.host,
        port=config.embedding_server.port,
    )


if __name__ == "__main__":
    main()
