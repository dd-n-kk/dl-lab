from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from functools import wraps
from typing import Any

__all__ = (
    "Callback",
    "Scheme",
)


# Subclass should call `super().__init__()`.
class Callback(ABC):
    def __init__(self) -> None:
        self._scheme: Scheme | None = None

    def __init_subclass__(cls) -> None:
        sub_register_scheme = cls.register_scheme

        @wraps(sub_register_scheme)
        def register_scheme(self, scheme: Scheme) -> None:
            sub_register_scheme(self, scheme)
            self._set_scheme(scheme)

        cls.register_scheme = register_scheme

    @abstractmethod
    def __call__(self) -> None: ...

    @property
    def scheme(self) -> Scheme | None:
        return self._scheme

    def register_scheme(self, scheme: Scheme) -> None:
        # Do not add `self._set_scheme(scheme)` here.
        # __init_subclass__() will add it even if subclass does not override this method.
        return

    def start_run(self) -> None:
        return

    def _set_scheme(self, scheme: Scheme) -> None:
        self._scheme = scheme


# Subclass should call `super().__init__()`.
class Scheme(ABC):
    hooks: tuple[str, ...]

    def __init__(self) -> None:
        self.callbacks: defaultdict[str, list[Callback]] = defaultdict(list)
        self.record: defaultdict[str, list[Any]] = defaultdict(list)

    def __init_subclass__(cls) -> None:
        sub_run = cls.run

        @wraps(sub_run)
        def run(self, *args, **kwargs) -> Any:
            self._start_run()
            return sub_run(self, *args, **kwargs)

        cls.run = run

    @abstractmethod
    def run(self, *args, **kwargs) -> Any: ...

    # Because Callback registration may depend on Scheme attributes,
    # this should be called at the end of __init__() or later.
    def register_callbacks(
        self, hook: str, callbacks: Callback | Iterable[Callback], *, reset: bool = False
    ) -> None:
        if reset:
            self.callbacks[hook].clear()

        if not isinstance(callbacks, Iterable):
            callbacks = (callbacks,)

        for callback in callbacks:
            callback.register_scheme(self)
            self.callbacks[hook].append(callback)

    def call_callbacks(self, hook: str) -> None:
        for callback in self.callbacks[hook]:
            callback()

    def clear_callbacks(self, hook: str | None = None) -> None:
        if hook is None:
            self.callbacks.clear()
        else:
            self.callbacks[hook].clear()

    def clear_record(self, key: str | None = None) -> None:
        if key is None:
            self.record.clear()
        else:
            self.record[key].clear()

    def _start_run(self):
        for hook, callbacks in self.callbacks.items():
            for callback in callbacks:
                if callback.scheme is not self:
                    raise RuntimeError(
                        f"{callback} of '{hook}' hook is not registered to this Scheme."
                    )
                callback.start_run()
