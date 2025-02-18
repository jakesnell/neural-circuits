import torch

from partikel.feynman_kac import FeynmanKac
from partikel.particle_set import ParticleSet
from partikel.resampler import Resampler, gather_ancestors


class ParticleFilter:
    def __init__(self, feynman_kac: FeynmanKac, resampler: Resampler):
        self.feynman_kac = feynman_kac
        self.resampler = resampler

    def run(self, num_particles: int) -> ParticleSet:
        sample_shape = torch.Size([num_particles])
        time_dim = len(sample_shape) + len(self.feynman_kac.batch_shape)

        log_potential, (loc, _) = (particle_set := self.feynman_kac.init(sample_shape))

        log_potentials = log_potential.unsqueeze(time_dim)
        xs = loc.unsqueeze(time_dim)

        for t in range(1, len(self.feynman_kac)):
            particle_set, ancestors = self.resampler.resample(particle_set)
            log_potentials = gather_ancestors(ancestors, log_potentials)
            xs = gather_ancestors(ancestors, xs)
            log_potential, (loc, _) = (
                particle_set := self.feynman_kac.step(t, particle_set)
            )
            log_potentials = torch.concatenate(
                [log_potentials, log_potential.unsqueeze(time_dim)], dim=time_dim
            )
            xs = torch.concatenate([xs, loc.unsqueeze(time_dim)], dim=time_dim)

        return ParticleSet(log_potentials, xs)
