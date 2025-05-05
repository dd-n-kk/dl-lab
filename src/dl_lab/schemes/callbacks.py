from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Self

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from dl_lab import utils

from .abc import Callback, Scheme

__all__ = (
    "RecordAttribute",
    "RecordLR",
    "RecordMovingAvg",
    "ScheduleLR",
    "TabulateLatest",
)


class RecordAttribute(Callback):
    def __init__(
        self,
        path: str,
        *,
        key: str | None = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        super().__init__()
        full_path = utils.attr_path(path)
        self.handle_path = full_path[:-1]
        self.attr_name = full_path[-1]
        self.transform = transform
        self.key = self.attr_name if key is None else key

        self.record: defaultdict[str, list[Any]]
        self.handle: object

    def __call__(self) -> None:
        attr = getattr(self.handle, self.attr_name)
        if self.transform is not None:
            attr = self.transform(attr)
        self.record[self.key].append(attr)

    @classmethod
    def one_item_array(cls, path: str, *, key: str | None = None) -> Self:
        return cls(path, key=key, transform=lambda x: x.item())

    def register_scheme(self, scheme: Scheme) -> None:
        self.record = scheme.record
        self.handle = utils.nested_getattr(scheme, self.handle_path)
        if self.transform is None:
            _ = getattr(self.handle, self.attr_name)
        else:
            _ = self.transform(getattr(self.handle, self.attr_name))


class RecordMovingAvg(Callback):
    def __init__(
        self, of_key: str, window: int | None = None, *, as_key: str | None = None
    ) -> None:
        super().__init__()
        self.of_key = of_key
        self.window = window
        if as_key is None:
            self.as_key = f"{of_key}_AVG" if window is None else f"{of_key}_MA{window}"
        else:
            self.as_key = as_key

    def __call__(self) -> None:
        values = self.record[self.of_key]
        if self.window is not None:
            ma = sum(values[-self.window :]) / self.window
        elif values:
            ma = sum(values) / len(values)
        else:
            ma = float("nan")
        self.record[self.as_key].append(ma)

    def register_scheme(self, scheme: Scheme) -> None:
        self.record = scheme.record


class RecordLR(Callback):
    def __init__(
        self, optimizer: Optimizer, *, key: str = "lr", param_group: int = 0
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.param_group = param_group
        self.key = key

        self.record: defaultdict[str, list[Any]]

    def __call__(self) -> None:
        self.record[self.key].append(self.optimizer.param_groups[self.param_group]["lr"])

    def register_scheme(self, scheme: Scheme) -> None:
        self.record = scheme.record
        _ = self.optimizer.param_groups[self.param_group]["lr"]


class ScheduleLR(Callback):
    def __init__(self, scheduler: LRScheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def __call__(self) -> None:
        self.scheduler.step()


class TabulateLatest(Callback):
    def __init__(
        self,
        keys: Iterable[str] | None = None,
        *,
        sort_keys: bool = False,
        decimals: int = 4,
        header_intvl: int = 10,  # 0: No header. <0: Only once. >0: 1 header per x rows.
        min_col_width: int = 16,
    ) -> None:
        super().__init__()
        self.keys = None if keys is None else tuple(sorted(keys) if sort_keys else keys)
        self.decimals = decimals
        self.header_intvl = header_intvl
        self.min_col_width = min_col_width

        self.count = 0

        self.record: defaultdict[str, list[Any]]

    def __call__(self) -> None:
        if (keys := self.keys) is None:
            keys = tuple(k for k in self.record if not k.startswith("_"))

        w = self.min_col_width

        row_vals = ((self.record[k][-1] if self.record[k] else None) for k in keys)
        row_str = " ".join(
            f"{v:>{w}.{self.decimals}g}" if isinstance(v, float) else f"{v:>{w}}"
            for v in row_vals
        )

        if (self.header_intvl < 0 and self.count == 0) or (
            self.header_intvl > 0 and self.count % self.header_intvl == 0
        ):
            header_str = " ".join(f"{k:>{w}}" for k in keys)
            tqdm.write(f"\n{header_str}\n{row_str}")
        else:
            tqdm.write(row_str)

        self.count += 1

    def register_scheme(self, scheme: Scheme) -> None:
        self.record = scheme.record

    def start_run(self) -> None:
        self.count = 0
