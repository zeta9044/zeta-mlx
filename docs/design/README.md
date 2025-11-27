# 함수형 디자인 원칙

Scott Wlaschin의 함수형 프로그래밍 패턴을 Python에 적용한 설계 문서입니다.

## 문서 목록

| 문서 | 설명 | 원본 자료 |
|------|------|----------|
| [functional-design.md](./functional-design.md) | FP 디자인 패턴 개요 | [NDC 2014](https://www.slideshare.net/slideshow/fp-patterns-ndc-london2014/42373281) |
| [railway-oriented-programming.md](./railway-oriented-programming.md) | 에러 처리 패턴 | [fsharpforfunandprofit.com/rop](https://fsharpforfunandprofit.com/rop/) |
| [domain-modeling.md](./domain-modeling.md) | 도메인 모델링 | [Domain Modeling Made Functional](https://pragprog.com/titles/swdddf/domain-modeling-made-functional/) |
| [composition-patterns.md](./composition-patterns.md) | 함수 합성 패턴 | [F# for fun and profit](https://fsharpforfunandprofit.com/) |

## 핵심 원칙 요약

### 1. Functions are Things
```python
# 함수를 값으로 다루기
process: Callable[[str], str] = lambda s: s.upper()
transform = partial(process_data, validator=my_validator)
```

### 2. Composition Everywhere
```python
# 파이프라인으로 합성
pipeline = pipe(validate, transform, format, send)
result = pipeline(input_data)
```

### 3. Types are not Classes
```python
# 데이터와 행동 분리
@dataclass(frozen=True)
class Message:
    role: Role
    content: str

def format_message(msg: Message) -> str: ...
```

### 4. Make Illegal States Unrepresentable
```python
# 타입으로 제약 표현
Role = Literal["system", "user", "assistant"]
Temperature = NewType('Temperature', float)  # 0.0 ~ 2.0
```

### 5. Railway Oriented Programming
```python
# 에러를 값으로 처리
Result = Success[T] | Failure[E]

result = (
    Railway.of(request)
    .bind(validate)
    .bind(process)
    .map(format)
    .unwrap()
)
```

## 패턴 빠른 참조

| 패턴 | 용도 | 예시 |
|------|------|------|
| `map` | 값 변환 | `result.map(lambda x: x * 2)` |
| `bind` | Result 체이닝 | `result.bind(validate)` |
| `pipe` | 함수 합성 | `pipe(f, g, h)(x)` |
| `partial` | 파라미터 고정 | `partial(generate, model="Qwen")` |
| `Monoid` | 값 결합 | `reduce(combine, items, empty)` |

## Scott Wlaschin 참고 자료

- [F# for fun and profit](https://fsharpforfunandprofit.com/) - 블로그
- [Domain Modeling Made Functional](https://pragprog.com/titles/swdddf/domain-modeling-made-functional/) - 책
- [Functional Programming Design Patterns](https://www.youtube.com/watch?v=srQt1NAHYC0) - NDC 강연
- [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/) - ROP 패턴
