from collections.abc import Iterator
from typing import Protocol

from torch import Tensor


class DataSource(Protocol):
    def __iter__(self) -> Iterator[tuple[Tensor, ...]]: ...

    def __len__(self) -> int: ...
