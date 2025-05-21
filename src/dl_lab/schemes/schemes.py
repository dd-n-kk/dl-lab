from collections.abc import Callable, Iterable
from typing import Any

import torch as tc
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from tqdm.auto import tqdm

from dl_lab import utils
from dl_lab.metrics import Metric
from dl_lab.types import DataSource

from .abc import Callback, Scheme

__all__ = ("SupervisedSGD",)


class SupervisedSGD(Scheme):
    hooks = ("per_batch", "per_epoch", "per_run")

    def __init__(
        self,
        *,
        data: DataSource,
        model: Module,
        optimizer: Optimizer,
        loss_func: Callable[..., Tensor],
        valid_data: DataSource | None = None,
        metrics: Iterable[Metric] = (),
        per_batch: Callback | Iterable[Callback] | None = None,
        per_epoch: Callback | Iterable[Callback] | None = None,
        per_run: Callback | Iterable[Callback] | None = None,
    ) -> None:
        super().__init__()

        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.valid_data = valid_data
        self.metrics = list(metrics)

        self.epoch_count = 0
        self.batch_count = 0
        self.loss = tc.tensor(float("nan"), device=utils.get_device())

        if per_batch:
            self.register_callbacks("per_batch", per_batch)
        if per_epoch:
            self.register_callbacks("per_epoch", per_epoch)
        if per_run:
            self.register_callbacks("per_run", per_run)

    def run(self, n_epochs: int) -> dict[str, Any]:
        with (
            tc.enable_grad(),
            utils.training(self.model),
            tqdm(total=len(self.data) * n_epochs) as progress_bar,
        ):
            device = utils.get_device()
            for _ in range(n_epochs):
                for batch in self.data:
                    *inputs, targets = (x.to(device) for x in batch)
                    self.loss = self.loss_func(self.model(*inputs), targets)
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()

                    self.batch_count += 1
                    self.call_callbacks("per_batch")
                    progress_bar.update(1)

                self.validate()
                self.epoch_count += 1
                self.call_callbacks("per_epoch")

            self.call_callbacks("per_run")

        return dict(self.record)

    def validate(self) -> None:
        if self.valid_data is None:
            return

        with tc.inference_mode(), utils.evaluating(self.model):
            device = utils.get_device()
            targets_buffer = []
            predicts_buffer = []
            for batch in self.valid_data:
                *inputs, targets = (x.to(device) for x in batch)
                targets_buffer.append(targets)
                predicts_buffer.append(self.model(*inputs))

            all_predicts = tc.cat(predicts_buffer)
            all_targets = tc.cat(targets_buffer)
            for metric in self.metrics:
                for k, v in metric(all_predicts, all_targets).items():
                    self.record[k].append(v)
