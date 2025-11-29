# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2025-11-29

### Added
- README.md for each package (PyPI documentation)
- PyPI metadata (homepage, repository, keywords, classifiers)

## [0.3.1] - 2025-11-29

### Added
- PyPI 패키지 배포 (zeta-mlx-core, zeta-mlx-inference, zeta-mlx-embedding, zeta-mlx-cli, zeta-mlx-rag, zeta-mlx-langchain)
- GitHub Release 자동화 (gh CLI)

### Changed
- 패키지 의존성을 PyPI 버전으로 변경 (배포용)

## [0.3.0] - 2025-11-29

### Added
- vLLM-compatible `/tokenize`, `/detokenize` endpoints
- `TokenizeRequest`, `TokenizeResponse`, `DetokenizeRequest`, `DetokenizeResponse` DTO 추가
- `TokenizeResult`, `DetokenizeResult` 타입 추가

### Changed
- Engine에 `tokenize()`, `detokenize()` 메서드 추가
- API converters 리팩토링

## [0.2.0] - 2024

### Added
- README.md 문서 추가
- 한/영 혼용 임베딩 모델 지원 (BGE-M3, E5-Large)
- LLM/임베딩 서버 분리 설정 지원
- 임베딩 패키지 (mlx-llm-embedding) with functional design
- 임베딩 모델용 YAML 설정 지원
- CLI daemon 모드 지원

### Changed
- 네임스페이스 패키지 구조로 변경
- 패키지명 mlx-llm에서 zeta-mlx로 변경
- zeta-mlx-api를 zeta-mlx-inference로 통합
- serve 명령어를 llm으로 변경
- 서버당 단일 모델만 로드하도록 변경
- mlx-lm 0.28 API로 업데이트
- mlx, mlx-lm, langchain-core 최신 버전으로 업그레이드
- YAML 설정 및 멀티모델 지원

### Fixed
- namespace package 및 API 버그 수정
- CLI models list 사용법 메시지 수정 및 `__main__.py` 추가
- stream_generate 응답 처리 수정
- Qwen3 context_length 32768로 수정

## [0.1.0] - 2024

### Added
- Initial release
- Multi-package workspace with functional design patterns
- Scott Wlaschin functional patterns 기반 설계
- Functional design documentation
- MLX LLM Server 초기 구현

[Unreleased]: https://github.com/zeta9044/zeta-mlx/compare/0.3.2...HEAD
[0.3.2]: https://github.com/zeta9044/zeta-mlx/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/zeta9044/zeta-mlx/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/zeta9044/zeta-mlx/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/zeta9044/zeta-mlx/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/zeta9044/zeta-mlx/releases/tag/0.1.0
