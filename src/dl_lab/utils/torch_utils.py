__all__ = (
    "ask_for_cuda",
    "copy_to_like",
    "count_params",
    "evaluating",
    "training",
)


from types import TracebackType

import torch as tc
from torch import Tensor
from torch.nn import Module


def ask_for_cuda() -> tc.device:
    return tc.device("cuda" if tc.cuda.is_available() else "cpu")


def copy_to_like(src: Tensor, dest: Tensor) -> None:
    if src.shape != dest.shape:
        raise ValueError(f"Shape mismatch: From {src.shape} to {dest.shape}.")
    with tc.no_grad():
        dest.copy_(src)


def count_params(module: Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class evaluating:
    def __init__(self, model: Module) -> None:
        self.model = model
        self.was_training = model.training

    def __enter__(self) -> None:
        if self.was_training:
            self.model.train(False)

    def __exit__(
        self,
        except_type: type[BaseException] | None,
        except_val: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.was_training:
            self.model.train(True)


class training:
    def __init__(self, model: Module) -> None:
        self.model = model
        self.was_evaluating = not model.training

    def __enter__(self) -> None:
        if self.was_evaluating:
            self.model.train(True)

    def __exit__(
        self,
        except_type: type[BaseException] | None,
        except_val: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.was_evaluating:
            self.model.train(False)
