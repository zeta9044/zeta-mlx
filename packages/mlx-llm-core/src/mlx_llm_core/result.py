"""Result 타입과 Railway 패턴"""
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Union

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')
E2 = TypeVar('E2')


# ============================================================
# Result Type (OR Type)
# ============================================================

@dataclass(frozen=True)
class Success(Generic[T]):
    """성공 트랙"""
    value: T

    def __repr__(self) -> str:
        return f"Success({self.value!r})"


@dataclass(frozen=True)
class Failure(Generic[E]):
    """실패 트랙"""
    error: E

    def __repr__(self) -> str:
        return f"Failure({self.error!r})"


Result = Union[Success[T], Failure[E]]


# ============================================================
# Result 연산 (순수 함수)
# ============================================================

def map_result(
    result: Result[T, E],
    f: Callable[[T], U]
) -> Result[U, E]:
    """Success 값에 함수 적용 (Functor)"""
    match result:
        case Success(value):
            return Success(f(value))
        case Failure() as err:
            return err


def bind(
    result: Result[T, E],
    f: Callable[[T], Result[U, E]]
) -> Result[U, E]:
    """Result 반환 함수 체이닝 (Monad)"""
    match result:
        case Success(value):
            return f(value)
        case Failure() as err:
            return err


def map_error(
    result: Result[T, E],
    f: Callable[[E], E2]
) -> Result[T, E2]:
    """Failure 에러 변환"""
    match result:
        case Success() as ok:
            return ok
        case Failure(error):
            return Failure(f(error))


def tee(
    result: Result[T, E],
    f: Callable[[T], None]
) -> Result[T, E]:
    """부수효과 실행 (로깅 등)"""
    match result:
        case Success(value):
            f(value)
    return result


def unwrap_or(result: Result[T, E], default: T) -> T:
    """값 추출 또는 기본값"""
    match result:
        case Success(value):
            return value
        case Failure():
            return default


def unwrap_or_else(result: Result[T, E], f: Callable[[E], T]) -> T:
    """값 추출 또는 에러로부터 계산"""
    match result:
        case Success(value):
            return value
        case Failure(error):
            return f(error)


# ============================================================
# Railway 파이프라인 빌더
# ============================================================

class Railway(Generic[T, E]):
    """Fluent Railway 파이프라인"""

    def __init__(self, result: Result[T, E]):
        self._result = result

    @classmethod
    def of(cls, value: T) -> 'Railway[T, E]':
        """Success로 시작"""
        return cls(Success(value))

    @classmethod
    def fail(cls, error: E) -> 'Railway[T, E]':
        """Failure로 시작"""
        return cls(Failure(error))

    @classmethod
    def from_result(cls, result: Result[T, E]) -> 'Railway[T, E]':
        """Result에서 생성"""
        return cls(result)

    def map(self, f: Callable[[T], U]) -> 'Railway[U, E]':
        """값 변환"""
        return Railway(map_result(self._result, f))

    def bind(self, f: Callable[[T], Result[U, E]]) -> 'Railway[U, E]':
        """Result 반환 함수 체이닝"""
        return Railway(bind(self._result, f))

    def tee(self, f: Callable[[T], None]) -> 'Railway[T, E]':
        """부수효과 실행"""
        return Railway(tee(self._result, f))

    def map_error(self, f: Callable[[E], E2]) -> 'Railway[T, E2]':
        """에러 변환"""
        return Railway(map_error(self._result, f))

    def recover(self, f: Callable[[E], Result[T, E]]) -> 'Railway[T, E]':
        """에러 복구 시도"""
        match self._result:
            case Success():
                return self
            case Failure(error):
                return Railway(f(error))

    def unwrap(self) -> Result[T, E]:
        """최종 Result 반환"""
        return self._result

    def unwrap_or(self, default: T) -> T:
        """값 또는 기본값"""
        return unwrap_or(self._result, default)

    def unwrap_or_raise(self, exception_fn: Callable[[E], Exception] | None = None) -> T:
        """값 또는 예외"""
        match self._result:
            case Success(value):
                return value
            case Failure(error):
                if exception_fn:
                    raise exception_fn(error)
                raise ValueError(str(error))


# ============================================================
# 병렬 검증 (Applicative)
# ============================================================

def validate_all(*results: Result[T, E]) -> Result[list[T], list[E]]:
    """모든 검증 실행, 에러 누적"""
    successes: list[T] = []
    failures: list[E] = []

    for r in results:
        match r:
            case Success(v):
                successes.append(v)
            case Failure(e):
                failures.append(e)

    if failures:
        return Failure(failures)
    return Success(successes)
