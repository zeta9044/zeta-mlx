"""함수 합성 유틸리티"""
from typing import TypeVar, Callable, Any
from functools import reduce

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def pipe(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    """왼쪽에서 오른쪽으로 함수 합성"""
    def apply(x: Any) -> Any:
        return reduce(lambda acc, f: f(acc), funcs, x)
    return apply


def compose(*funcs: Callable[..., Any]) -> Callable[..., Any]:
    """오른쪽에서 왼쪽으로 함수 합성"""
    return pipe(*reversed(funcs))


def identity(x: A) -> A:
    """항등 함수"""
    return x


def const(value: A) -> Callable[[B], A]:
    """상수 함수"""
    return lambda _: value


def curry2(f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    """2인자 함수 커링"""
    return lambda a: lambda b: f(a, b)


def flip(f: Callable[[A, B], C]) -> Callable[[B, A], C]:
    """인자 순서 뒤집기"""
    return lambda b, a: f(a, b)
