from collections.abc import Callable
from typing import Literal

from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy


def FlatteningBCELoss(
    *, reduction: Literal["mean", "sum"] = "mean", pos_weight: Tensor | None = None
) -> Callable[[Tensor, Tensor], Tensor]:
    def flattening_bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
        return binary_cross_entropy_with_logits(
            logits.flatten(),
            targets.flatten().float(),
            reduction=reduction,
            pos_weight=pos_weight,
        )

    return flattening_bce_loss


def FlatteningCELoss(
    *,
    axis: int = -1,
    reduction: Literal["mean", "sum"] = "mean",
    weight: Tensor | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    if axis == -1:

        def flattening_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
            return cross_entropy(
                logits.flatten(0, -2),
                targets.flatten(),
                reduction=reduction,
                weight=weight,
            )
    else:

        def flattening_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
            return cross_entropy(
                logits.moveaxis(axis, -1).flatten(0, -2),
                targets.flatten(),
                reduction=reduction,
                weight=weight,
            )

    return flattening_ce_loss
