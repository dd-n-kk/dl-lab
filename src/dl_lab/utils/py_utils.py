__all__ = (
    "attr_path",
    "nested_getattr",
)


from collections.abc import Iterable
from functools import reduce
from typing import Any


def attr_path(spec: str) -> tuple[str, ...]:
    return tuple(spec.split("."))


def nested_getattr(obj: object, path: Iterable[str]) -> Any:
    return reduce(getattr, path, obj)
