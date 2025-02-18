import abc
from functools import partial
import torch
import tree
from partikel.particle_set import ParticleSet

from tree import map_structure


def ancestor_to(ancestors, target):
    return ancestors.view(
        ancestors.shape + torch.Size([1] * (target.ndim - ancestors.ndim))
    ).expand(ancestors.shape + target.shape[ancestors.ndim :])


def gather_ancestors(ancestors, x):
    return x.gather(0, ancestor_to(ancestors, x))


class Resampler(abc.ABC):
    @abc.abstractmethod
    def resample(self, p: ParticleSet) -> tuple[ParticleSet, torch.Tensor]:
        """Resample the particles according to their weights"""


class MultinomialResampler(Resampler):
    def resample(self, p: ParticleSet) -> tuple[ParticleSet, torch.Tensor]:
        permute_order = list(range(p.weights.ndim))
        permute_order = permute_order[1:] + [0]

        ancestors = torch.multinomial(
            p.weights.permute(permute_order), p.num_particles, replacement=True
        )

        reverse_order = list(range(p.weights.ndim))
        reverse_order = [reverse_order[-1]] + reverse_order[:-1]
        ancestors = ancestors.permute(reverse_order)

        return (
            ParticleSet(
                logits=torch.zeros_like(p.logits),
                particles=tree.map_structure(
                    partial(gather_ancestors, ancestors), p.particles
                ),
            ),
            ancestors,
        )


class AdaptiveResampler(Resampler):
    def __init__(self, base_resampler: Resampler, ratio: float):
        assert 0 <= ratio <= 1.0, "ratio must be between 0 and 1"
        self.base_resampler = base_resampler
        self.ratio = ratio

    def resample(self, p: ParticleSet) -> tuple[ParticleSet, torch.Tensor]:
        resample_mask = p.ess < self.ratio * p.num_particles

        resampled_p, resampled_ancestors = self.base_resampler.resample(p)

        def apply_mask(if_true: torch.Tensor, if_false: torch.Tensor):
            assert if_true.shape == if_false.shape
            m = resample_mask.unsqueeze(0).expand(
                if_true.shape[: resample_mask.ndim + 1]
            )
            return torch.where(ancestor_to(m, if_true), if_true, if_false)

        return map_structure(apply_mask, resampled_p, p), map_structure(
            apply_mask,
            resampled_ancestors,
            ancestor_to(
                torch.arange(p.num_particles, device=p.weights.device), p.weights
            ),
        )
