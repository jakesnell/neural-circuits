import abc
import torch

from attrs import define


@define
class Stats(abc.ABC):
    pass


@define
class MarkovStats(Stats):
    loc: torch.Tensor
