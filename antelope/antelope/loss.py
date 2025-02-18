import torch
import torch.nn.functional as F


def apply_reduction(x: torch.Tensor, reduction: str, axis: int):
    if reduction == "none":
        return x
    elif reduction == "sum":
        return torch.sum(x, axis=axis)
    elif reduction == "mean":
        return torch.mean(x, axis=axis)
    else:
        raise ValueError(f"unexpected reduction {reduction}")


class BatchedCrossEntropy:
    def __init__(self):
        self.loss_fun = torch.vmap(
            lambda input, target: F.cross_entropy(input, target, reduction="none"),
            in_dims=(0, 0),
        )

    def __call__(self, input, target, batch_reduction="mean", time_reduction="sum"):
        loss_val = self.loss_fun(input, target)

        loss_val = apply_reduction(loss_val, reduction=batch_reduction, axis=-2)
        loss_val = apply_reduction(loss_val, reduction=time_reduction, axis=-1)

        return loss_val
