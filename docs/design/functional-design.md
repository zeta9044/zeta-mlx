# Scott Wlaschin 함수형 디자인 패턴

Scott Wlaschin의 함수형 프로그래밍 디자인 패턴을 Python에 적용합니다.

> 참고: [Functional Programming Design Patterns (NDC London 2014)](https://www.slideshare.net/slideshow/fp-patterns-ndc-london2014/42373281)

## 핵심 원칙 3가지

### 1. Functions are things

함수는 일급 시민(first-class citizen)입니다. 변수에 할당하고, 파라미터로 전달하고, 반환값으로 사용할 수 있습니다.

```python
# 함수를 값으로 다루기
from typing import Callable

# 함수 타입 정의
Predicate = Callable[[int], bool]
Transform = Callable[[str], str]

# 함수를 파라미터로 전달
def filter_list(predicate: Predicate, items: list[int]) -> list[int]:
    return [x for x in items if predicate(x)]

# 함수를 반환
def make_adder(n: int) -> Callable[[int], int]:
    return lambda x: x + n

add_5 = make_adder(5)
result = add_5(10)  # 15
```

### 2. Composition everywhere

작은 함수들을 조합하여 큰 함수를 만듭니다.

```python
from typing import TypeVar, Callable

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

# 함수 합성 연산자
def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """g 먼저 실행 후 f 실행 (f ∘ g)"""
    return lambda x: f(g(x))

def pipe(f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]:
    """f 먼저 실행 후 g 실행 (f >> g)"""
    return lambda x: g(f(x))

# 파이프라인 빌더
def pipeline(*funcs):
    """여러 함수를 순차적으로 합성"""
    def apply(x):
        result = x
        for f in funcs:
            result = f(result)
        return result
    return apply

# 사용 예시
normalize = lambda s: s.strip().lower()
remove_spaces = lambda s: s.replace(" ", "_")
add_prefix = lambda s: f"mlx_{s}"

process_name = pipeline(normalize, remove_spaces, add_prefix)
result = process_name("  Hello World  ")  # "mlx_hello_world"
```

### 3. Types are not classes

타입은 데이터의 형태를 정의하며, 행동(behavior)은 별도의 함수로 분리합니다.

```python
from dataclasses import dataclass
from typing import Literal

# AND 타입 (Product Type) - 모든 필드가 필요
@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

# OR 타입 (Sum Type) - 여러 가능성 중 하나
@dataclass(frozen=True)
class Success[T]:
    value: T

@dataclass(frozen=True)
class Failure[E]:
    error: E

Result = Success[T] | Failure[E]

# 행동은 별도 함수로
def format_message(msg: Message) -> str:
    return f"[{msg.role}] {msg.content}"
```

## 패턴 1: Functions as Parameters

OOP의 Strategy 패턴을 함수로 대체합니다.

```python
# OOP Strategy 패턴 (복잡함)
class SortStrategy:
    def sort(self, data): pass

class QuickSort(SortStrategy):
    def sort(self, data): ...

class MergeSort(SortStrategy):
    def sort(self, data): ...

# FP 방식 (단순함)
from typing import Callable, TypeVar

T = TypeVar('T')
SortFn = Callable[[list[T]], list[T]]

def process_data(data: list[T], sort_fn: SortFn) -> list[T]:
    return sort_fn(data)

# 사용
result = process_data([3, 1, 2], sorted)
result = process_data([3, 1, 2], lambda x: sorted(x, reverse=True))
```

## 패턴 2: Partial Application

다중 파라미터 함수를 단계별로 적용합니다.

```python
from functools import partial

def generate(model: str, temperature: float, prompt: str) -> str:
    """3개 파라미터를 받는 함수"""
    return f"[{model}@{temperature}] {prompt}"

# Partial Application으로 파라미터 고정
generate_with_qwen = partial(generate, "Qwen3-8B")
generate_creative = partial(generate_with_qwen, 0.9)

# 최종 호출
result = generate_creative("Hello!")  # "[Qwen3-8B@0.9] Hello!"
```

## 패턴 3: Functor (Map)

컨테이너 안의 값을 변환합니다.

```python
from typing import TypeVar, Callable, Generic
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')

@dataclass(frozen=True)
class Box(Generic[T]):
    """값을 담는 컨테이너 (Functor)"""
    value: T

    def map(self, f: Callable[[T], U]) -> 'Box[U]':
        """내부 값에 함수 적용"""
        return Box(f(self.value))

# 사용
box = Box(5)
result = box.map(lambda x: x * 2).map(lambda x: x + 1)  # Box(11)

# Result도 Functor
def map_result(result: Result[T, E], f: Callable[[T], U]) -> Result[U, E]:
    match result:
        case Success(value):
            return Success(f(value))
        case Failure() as err:
            return err
```

## 패턴 4: Monad (Bind)

효과가 있는 함수들을 체이닝합니다.

```python
from typing import TypeVar, Callable

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')

def bind(
    result: Result[T, E],
    f: Callable[[T], Result[U, E]]
) -> Result[U, E]:
    """Result를 반환하는 함수들을 체이닝"""
    match result:
        case Success(value):
            return f(value)
        case Failure() as err:
            return err

# 사용 예시
def parse_int(s: str) -> Result[int, str]:
    try:
        return Success(int(s))
    except ValueError:
        return Failure(f"'{s}'는 숫자가 아닙니다")

def validate_positive(n: int) -> Result[int, str]:
    if n > 0:
        return Success(n)
    return Failure(f"{n}은 양수가 아닙니다")

# 체이닝
result = bind(parse_int("42"), validate_positive)  # Success(42)
result = bind(parse_int("-5"), validate_positive)  # Failure("-5은 양수가 아닙니다")
result = bind(parse_int("abc"), validate_positive)  # Failure("'abc'는 숫자가 아닙니다")
```

## 패턴 5: Monoid

같은 타입의 것들을 결합합니다.

```python
from typing import TypeVar, Protocol, Callable
from dataclasses import dataclass

T = TypeVar('T')

class Monoid(Protocol[T]):
    """Monoid 프로토콜"""
    @staticmethod
    def empty() -> T: ...
    @staticmethod
    def combine(a: T, b: T) -> T: ...

# 규칙:
# 1. Closure: combine(a, b) -> 같은 타입
# 2. Associativity: combine(combine(a, b), c) == combine(a, combine(b, c))
# 3. Identity: combine(a, empty()) == a

# 문자열 Monoid
class StringMonoid:
    @staticmethod
    def empty() -> str:
        return ""

    @staticmethod
    def combine(a: str, b: str) -> str:
        return a + b

# 리스트 Monoid
class ListMonoid:
    @staticmethod
    def empty() -> list:
        return []

    @staticmethod
    def combine(a: list, b: list) -> list:
        return a + b

# Monoid로 reduce
def concat_all(monoid: type, items: list[T]) -> T:
    result = monoid.empty()
    for item in items:
        result = monoid.combine(result, item)
    return result

# 사용
messages = ["Hello", " ", "World"]
result = concat_all(StringMonoid, messages)  # "Hello World"
```

## OOP 패턴 vs FP 패턴

| OOP 패턴 | FP 대체 |
|----------|---------|
| Strategy | 함수를 파라미터로 전달 |
| Decorator | 함수 합성 |
| Visitor | 패턴 매칭 + 재귀 |
| Factory | 함수 반환 |
| Singleton | 모듈 레벨 값 |
| Observer | 콜백 함수 리스트 |

## SOLID 원칙의 FP 해석

| SOLID | FP 해석 |
|-------|---------|
| **S**ingle Responsibility | 함수는 하나의 일만 |
| **O**pen/Closed | 합성으로 확장, 수정 없음 |
| **L**iskov Substitution | 타입 시그니처 준수 |
| **I**nterface Segregation | 작은 함수 타입들 |
| **D**ependency Inversion | 함수를 파라미터로 주입 |

## 참고 자료

- [F# for fun and profit](https://fsharpforfunandprofit.com/)
- [Functional Programming Design Patterns - NDC 2014](https://www.slideshare.net/slideshow/fp-patterns-ndc-london2014/42373281)
- [DevTernity 2018 Slides](https://www.slideshare.net/ScottWlaschin/functional-design-patterns-devternity2018)
