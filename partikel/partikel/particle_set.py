import math
from functools import partial
from typing import NamedTuple, Union, Callable
import torch
import tree
from tree import Structure


class ParticleSet(NamedTuple):
    logits: torch.Tensor  # unnormalized log weights
    particles: Union[torch.Tensor, Structure]

    @property
    def num_particles(self):
        return self.logits.size(0)

    @property
    def weights(self):
        return torch.softmax(self.logits, dim=0)

    @property
    def log_evidence(self):
        return torch.logsumexp(self.logits, dim=0) - math.log(self.logits.size(0))

    @property
    def ess(self):
        return torch.reciprocal(torch.sum(torch.square(self.weights), dim=0))

    @staticmethod
    def apply_weights(W, x):
        W_broadcast = W.view(*W.shape, *(1,) * (x.ndim - W.ndim))
        return torch.sum(W_broadcast * x, dim=0)

    def estimate(self):
        """Return the estimate by weighting particles according to logits"""
        return tree.map_structure(
            partial(self.apply_weights, self.weights), self.particles
        )

    def estimate_func(self, f: Callable):
        ret = 0.0
        W = self.weights
        assert W.ndim == 2
        W = W[:, -1]
        for i in range(self.num_particles):
            Wi = W[i].item()
            ret += f(self.particles[i]) * Wi
        return ret
