import torch
from torch.distributions import Distribution, Normal
from pyro.distributions import Delta

from partikel.stats import MarkovStats


class DeltaTransition:
    def __call__(self, x_prev: torch.Tensor) -> Distribution:
        return Delta(x_prev)


class LinearGaussianTransition:
    def __init__(self, rho: float, sigma_x: float):
        self.rho = rho
        self.sigma_x = sigma_x

    def __call__(self, stats: MarkovStats) -> Distribution:
        return Normal(self.rho * stats.loc, self.sigma_x**2)


# @njit
# def _finite_sample(p):
#     n = p.shape[0]
#     k = p.shape[1]

#     u = np.random.rand(n)
#     x = np.empty(n, dtype=np.int64)
#     for i in range(n):
#         acc = 0.0
#         for j in range(k):
#             acc += p[i, j]
#             if acc >= u[i]:
#                 x[i] = j
#                 break

#     return x


# class FiniteMarkovKernel(MarkovKernel):
#     def __init__(self, A: torch.Tensor):
#         """Each row of A represents the transition probabilities to the next
#         state."""
#         assert A.ndim == 2
#         assert_close(A.sum(-1), torch.ones(A.size(0)))
#         self.A = A

#     def sample(self, x_prev: torch.Tensor) -> torch.Tensor:
#         p = self.A[x_prev]
#         return torch.from_numpy(_finite_sample(p.numpy()))
