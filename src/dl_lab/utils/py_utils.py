from collections.abc import Iterable
from functools import reduce
from typing import Any

__all__ = (
    "attr_path",
    "nested_getattr",
)


def attr_path(spec: str) -> tuple[str, ...]:
    return tuple(spec.split("."))


def nested_getattr(obj: object, path: Iterable[str]) -> Any:
    return reduce(getattr, path, obj)
