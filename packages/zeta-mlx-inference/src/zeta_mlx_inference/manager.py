"""다중 모델 관리자 (LRU 기반)"""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator
from threading import Lock

from zeta_mlx_core import (
    Result, Success, Failure,
    ModelsConfig, ModelDefinition,
    ModelNotFoundError,
    Message, GenerationParams,
)
from zeta_mlx_inference.loader import load_model_safe
from zeta_mlx_inference.engine import InferenceEngine


@dataclass
class LoadedModel:
    """로드된 모델 정보"""
    alias: str
    definition: ModelDefinition
    engine: InferenceEngine


class ModelManager:
    """
    다중 모델 관리자

    LRU 방식으로 모델을 관리하며, 최대 로드 수를 초과하면
    가장 오래 사용되지 않은 모델을 언로드합니다.
    """

    def __init__(self, config: ModelsConfig):
        self._config = config
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = Lock()

    @property
    def default_alias(self) -> str:
        """기본 모델 별칭"""
        return self._config.default

    def list_available(self) -> list[str]:
        """사용 가능한 모델 별칭 목록"""
        return self._config.list_aliases()

    def list_loaded(self) -> list[str]:
        """현재 로드된 모델 별칭 목록"""
        with self._lock:
            return list(self._loaded.keys())

    def get_model_info(self, alias: str) -> ModelDefinition | None:
        """모델 정의 조회"""
        return self._config.get_model(alias)

    def resolve_alias(self, model_name: str) -> str:
        """
        모델 이름을 별칭으로 해석

        - 빈 문자열 또는 None → 기본 모델
        - 별칭이 존재하면 그대로 사용
        - HuggingFace 경로면 역방향 조회
        """
        if not model_name:
            return self._config.default

        # 별칭으로 직접 존재하는지
        if model_name in self._config.available:
            return model_name

        # HuggingFace 경로로 역방향 조회
        for alias, defn in self._config.available.items():
            if defn.path == model_name:
                return alias

        # 찾지 못하면 그대로 반환 (에러는 호출자가 처리)
        return model_name

    def get_engine(self, alias: str) -> Result[InferenceEngine, ModelNotFoundError]:
        """
        모델 엔진 가져오기 (필요시 로드)

        LRU 방식으로 최근 사용된 모델을 유지합니다.
        """
        resolved = self.resolve_alias(alias)

        with self._lock:
            # 이미 로드된 경우: LRU 갱신
            if resolved in self._loaded:
                self._loaded.move_to_end(resolved)
                return Success(self._loaded[resolved].engine)

            # 모델 정의 확인
            definition = self._config.get_model(resolved)
            if definition is None:
                return Failure(ModelNotFoundError(model_name=resolved))

            # 최대 로드 수 초과 시 가장 오래된 모델 언로드
            while len(self._loaded) >= self._config.max_loaded:
                oldest_alias, oldest = self._loaded.popitem(last=False)
                print(f"Unloading model: {oldest_alias}")
                del oldest  # GC에서 메모리 해제

            # 모델 로드
            print(f"Loading model: {resolved} ({definition.path})")
            bundle_result = load_model_safe(definition.path)

            if isinstance(bundle_result, Failure):
                return Failure(ModelNotFoundError(model_name=definition.path))

            engine = InferenceEngine(bundle_result.value)

            self._loaded[resolved] = LoadedModel(
                alias=resolved,
                definition=definition,
                engine=engine,
            )

            return Success(engine)

    def generate(
        self,
        alias: str,
        messages: list[Message],
        params: GenerationParams,
    ):
        """지정된 모델로 생성"""
        engine_result = self.get_engine(alias)

        if isinstance(engine_result, Failure):
            return engine_result

        from zeta_mlx_core import ChatRequest
        request = ChatRequest(
            model=alias,
            messages=messages,
            params=params,
            stream=False,
        )
        return engine_result.value.generate(request)

    def stream(
        self,
        alias: str,
        messages: list[Message],
        params: GenerationParams,
    ) -> Iterator[str]:
        """지정된 모델로 스트리밍 생성"""
        engine_result = self.get_engine(alias)

        if isinstance(engine_result, Failure):
            yield f"[Error: Model '{alias}' not found]"
            return

        from zeta_mlx_core import ChatRequest
        request = ChatRequest(
            model=alias,
            messages=messages,
            params=params,
            stream=True,
        )
        yield from engine_result.value.stream(request)

    def preload(self, aliases: list[str]) -> dict[str, bool]:
        """지정된 모델들 미리 로드"""
        results = {}
        for alias in aliases:
            result = self.get_engine(alias)
            results[alias] = isinstance(result, Success)
        return results

    def unload(self, alias: str) -> bool:
        """모델 명시적 언로드"""
        with self._lock:
            if alias in self._loaded:
                del self._loaded[alias]
                return True
            return False

    def unload_all(self) -> None:
        """모든 모델 언로드"""
        with self._lock:
            self._loaded.clear()


# ============================================================
# 팩토리 함수
# ============================================================

def create_model_manager(config: ModelsConfig) -> ModelManager:
    """모델 관리자 생성"""
    return ModelManager(config)


def create_model_manager_from_yaml(config_path: str) -> Result[ModelManager, str]:
    """YAML 설정에서 모델 관리자 생성"""
    from zeta_mlx_core import load_config

    config_result = load_config(config_path)
    if isinstance(config_result, Failure):
        return Failure(str(config_result.error))

    return Success(ModelManager(config_result.value.models))
