from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from itertools import chain
from math import isfinite
from typing import Literal

import torch as tc
from torch import Tensor

__all__ = (
    "Metric",
    "AsMetric",
    "BasicBinaryMetrics",
    "BasicMulticlassMetrics",
)


class Metric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> dict[str, float]: ...

    @abstractmethod
    def names(self) -> tuple[str, ...]: ...


# `func` is expected to output scalar|1D Tensor.
class AsMetric(Metric):
    def __init__(self, func: Callable[..., Tensor], names: str | Iterable[str]) -> None:
        self.func = func
        self._names = (names,) if isinstance(names, str) else tuple(names)

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        output = self.func(*args, **kwargs)
        if output.numel() == 1:
            return {self._names[0]: output.item()}
        return dict(zip(self._names, output.tolist(), strict=True))


class BasicBinaryMetrics(Metric):
    def __init__(
        self,
        *,
        prediction: Literal["logit", "prob", "index"] = "logit",
        prefix: str = "",
        zero_div_val: Literal["nan", "0", "1"] = "nan",
    ) -> None:
        self.threshold = (
            0.0 if prediction == "logit" else 0.5 if prediction == "prob" else None
        )
        self.zero_div_val = float(zero_div_val)
        self._names = ("ACC", "F1", "TPR", "TNR", "PPV", "NPV")
        if prefix:
            self._names = tuple(f"{prefix}_{name}" for name in self._names)

    def __call__(self, predicts: Tensor, targets: Tensor) -> dict[str, float]:
        predicts = predicts.flatten()
        targets = targets.flatten()

        if self.threshold is not None:
            predicts = (predicts > self.threshold).long()

        n = targets.numel()
        calls = predicts.sum().item()
        counts = targets.sum().item()

        matches = predicts.eq(targets)
        true_pos = predicts.eq(1).logical_and(matches).sum().item()
        true_neg = n - calls - counts + true_pos

        acc = matches.sum().item() / n
        f1 = true_pos * 2 / (calls + counts) if calls + counts > 0 else self.zero_div_val
        tpr = true_pos / counts if counts > 0 else self.zero_div_val
        tnr = true_neg / (n - counts) if counts < n else self.zero_div_val
        ppv = true_pos / calls if calls > 0 else self.zero_div_val
        npv = true_neg / (n - calls) if calls < n else self.zero_div_val

        return dict(zip(self._names, (acc, f1, tpr, tnr, ppv, npv), strict=True))

    def names(self) -> tuple[str, ...]:
        return self._names


class BasicMulticlassMetrics(Metric):
    def __init__(
        self,
        n_classes: int,
        *,
        axis: int = -1,
        prediction: Literal["logit", "prob", "index"] = "logit",
        prefix: str = "",
        labels: Iterable[str] | None = None,
        zero_div_val: Literal["nan", "0", "1"] | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.axis = axis
        self.prediction = prediction
        self.zero_div_val = None if zero_div_val is None else float(zero_div_val)
        self._names = self._make_names(n_classes, prefix, labels)

    def __call__(self, predicts: Tensor, targets: Tensor) -> dict[str, float]:
        if self.prediction == "index":
            predicts = predicts.flatten()
        else:
            if self.axis != -1:
                predicts = predicts.moveaxis(self.axis, -1)
            predicts = predicts.argmax(-1).flatten()

        targets = targets.flatten()

        n = targets.numel()
        c = self.n_classes
        calls = predicts.bincount(minlength=c)
        counts = targets.bincount(minlength=c)

        classes = tc.arange(c)
        matches = predicts.eq(targets)
        true_pos = predicts.eq(classes.unsqueeze(-1)).logical_and(matches).sum(-1)
        true_neg = n - calls - counts + true_pos

        acc = (matches.sum().item() / n,)
        f1 = (true_pos * 2 / (calls + counts)).tolist()
        tpr = (true_pos / counts).tolist()
        tnr = (true_neg / (n - counts)).tolist()
        ppv = (true_pos / calls).tolist()
        npv = (true_neg / (n - calls)).tolist()

        results = dict(zip(self._names, chain(acc, f1, tpr, tnr, ppv, npv), strict=True))

        if self.zero_div_val is not None:
            for k, v in results.items():
                if not isfinite(v):
                    results[k] = self.zero_div_val

        return results

    def names(self) -> tuple[str, ...]:
        return self._names

    @staticmethod
    def _make_names(
        n_classes: int, prefix: str = "", labels: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if labels:
            labels_ = tuple(labels)
            if len(labels_) != n_classes:
                raise ValueError(f"Got {len(labels_)} labels for {n_classes} classes.")
        else:
            labels_ = tuple(range(n_classes))

        names = (
            "ACC",
            *(f"F1_{label}" for label in labels_),
            *(f"TPR_{label}" for label in labels_),
            *(f"TNR_{label}" for label in labels_),
            *(f"PPV_{label}" for label in labels_),
            *(f"NPV_{label}" for label in labels_),
        )

        return tuple(f"{prefix}_{name}" for name in names) if prefix else names
