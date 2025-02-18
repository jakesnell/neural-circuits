import abc
from typing import Callable

from attrs import define

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical
from torch.distributions import constraints

from numba import njit

from partikel import ppl
from partikel.stats import Stats, MarkovStats


class MarkovProcess(abc.ABC):
    @abc.abstractmethod
    def init(self, sample_shape: torch.Size) -> tuple[torch.Tensor, Stats]:
        """Return x_0 and the initial stats"""

    @abc.abstractmethod
    def step(self, t: int, stats: Stats) -> tuple[torch.Tensor, Stats]:
        """Return x_t given stats"""

    def __call__(
        self, end: int, sample_shape: torch.Size = torch.Size([])
    ) -> tuple[torch.Tensor, Stats]:
        x, stats = self.init(sample_shape=sample_shape)

        xs = [x]

        for t in range(1, end + 1):
            x, stats = self.step(t, stats)
            xs.append(x)

        return torch.stack(xs, dim=len(sample_shape)), stats


class HomogeneousMarkovProcess(MarkovProcess):
    def __init__(self, initial: Distribution, transition_closure: Callable):
        self.initial = initial
        self.transition_closure = transition_closure

    def init(self, sample_shape: torch.Size) -> tuple[torch.Tensor, MarkovStats]:
        x_0 = ppl.sample("x_0", self.initial, sample_shape=sample_shape)
        return x_0, MarkovStats(x_0)

    def step(self, t: int, stats: MarkovStats) -> tuple[torch.Tensor, MarkovStats]:
        x_t = ppl.sample(f"x_{t}", self.transition_closure(stats))
        return x_t, MarkovStats(x_t)


@njit
def _crp_stats_observe_state(counts, n, k, x):
    num_particles = counts.shape[0]

    for i in range(num_particles):
        counts[i, x[i]] += 1
        n[i] += 1
        if x[i] >= k[i]:
            k[i] += 1


@define
class CRPStats(Stats):
    counts: torch.Tensor
    n: torch.Tensor
    k: torch.Tensor

    def clone(self):
        return CRPStats(self.counts.clone(), self.n.clone(), self.k.clone())


@njit
def _crp_sample(alpha, counts, n, k):
    num_particles = counts.shape[0]

    u = np.random.rand(num_particles) * (n + alpha)
    x = np.empty_like(n)
    for i in range(num_particles):
        acc = 0
        x[i] = k[i]
        for j in range(k[i]):
            acc += counts[i, j]
            if acc >= u[i]:
                x[i] = j
                break

    return x


class CRPCategorical(Distribution):
    arg_constraints = {}  # pyright: ignore

    def __init__(self, alpha: float, stats: CRPStats):
        stats = stats.clone()
        self.alpha = alpha
        self.counts = stats.counts
        self.n = stats.n
        self.k = stats.k

    @property
    def batch_shape(self):
        return self.counts.size()[:-1]

    @property
    def event_shape(self):
        return ()

    @property
    def mode(self):
        max_count, ind = self.counts.max(-1)
        return torch.where(max_count >= self.alpha, ind, self.k)

    def logits(self, num_classes: int):
        padded_counts = torch.zeros(
            self.n.shape + torch.Size([num_classes]), device=self.counts.device
        )
        padded_counts[..., : self.counts.size(-1)] = self.counts
        flat_padded_counts = padded_counts.view(-1, num_classes)
        flat_padded_counts[
            torch.arange(flat_padded_counts.size(0)), self.k.view(-1)
        ] = self.alpha
        return torch.log(flat_padded_counts.view(padded_counts.shape))

    def sample(self) -> torch.Tensor:
        d = self.counts.size(-1)
        return (
            torch.from_numpy(
                _crp_sample(
                    self.alpha,
                    self.counts.view(-1, d).numpy(),
                    self.n.view(-1).numpy(),
                    self.k.view(-1).numpy(),
                )
            )
            .reshape_as(self.n)
            .to(self.n.device)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # in order to evaluate, need to ensure that x is broadcastable
        _, counts = torch.broadcast_tensors(x.unsqueeze(-1), self.counts)
        weights = torch.full(x.shape, self.alpha, device=counts.device)
        in_bounds = torch.nonzero(x < self.k, as_tuple=True)
        weights[in_bounds] = (
            torch.gather(counts[in_bounds], -1, x[in_bounds].unsqueeze(-1))
            .squeeze(-1)
            .to(weights.dtype)
        )
        out_of_bounds = torch.nonzero(x > self.k, as_tuple=True)
        weights[out_of_bounds] = 0.0

        return torch.log(weights) - torch.log(self.n + self.alpha)

    def enumerate_support(self, expand: bool = True) -> torch.Tensor:
        # note: for ordinary categorical, if logits are of shape
        # (..., K) then enumerate_support will return a tensor
        # of shape (K, ...)
        support = torch.arange(int(self.k.max()) + 1)
        support = support.reshape(
            support.size(0), *(1 for _ in range(self.counts.ndim - 1))
        )
        if expand:
            return support.expand(-1, *self.counts.size()[:-1])
        else:
            return support


class ChineseRestaurantProcess(MarkovProcess):
    def __init__(self, alpha: float, cuda: bool = torch.cuda.is_available()):
        self.alpha = alpha
        self.cuda = cuda

    def init(self, sample_shape: torch.Size) -> tuple[torch.Tensor, CRPStats]:
        x_0 = ppl.sample(
            "x_0",
            Categorical(logits=torch.zeros(1, device="cuda" if self.cuda else "cpu")),
            sample_shape=sample_shape,
        )

        assert torch.all(x_0 == 0), "can only initialize counts when state is all 0"
        counts = F.one_hot(x_0)
        n = counts.sum(-1)
        k = torch.ones_like(x_0)
        return x_0, CRPStats(counts, n, k)

    def predictive_distribution(self, stats: CRPStats) -> CRPCategorical:
        return CRPCategorical(self.alpha, stats)

    def step(self, t: int, stats: CRPStats) -> tuple[torch.Tensor, Stats]:
        stats = stats.clone()

        x_t = ppl.sample(f"x_{t}", CRPCategorical(self.alpha, stats))

        if torch.any(x_t >= stats.counts.size(-1)):
            stats.counts = F.pad(stats.counts, pad=(0, stats.counts.size(-1)))

        d = stats.counts.size(-1)

        counts_cpu = stats.counts.cpu()
        n_cpu = stats.n.cpu()
        k_cpu = stats.k.cpu()

        _crp_stats_observe_state(
            counts_cpu.view(-1, d).numpy(),
            n_cpu.view(-1).numpy(),
            k_cpu.view(-1).numpy(),
            x_t.cpu().view(-1).numpy(),
        )

        stats.counts = counts_cpu.to(stats.counts.device)
        stats.n = n_cpu.to(stats.n.device)
        stats.k = k_cpu.to(stats.k.device)

        return x_t, stats
