import abc
from collections.abc import Sized
import torch

from partikel import ppl
from partikel.particle_set import ParticleSet
from partikel.state_space_model import StateSpaceModel


class FeynmanKac(abc.ABC, Sized):
    @abc.abstractproperty
    def batch_shape(self) -> torch.Size:  # pyright: ignore
        """Return the batch shape"""

    @abc.abstractmethod
    def init(self, sample_shape: torch.Size) -> ParticleSet:
        """Return initial loc, initial log_potential, and initial state"""

    @abc.abstractmethod
    def step(self, t: int, particle_set: ParticleSet) -> ParticleSet:
        """Return the next particle set"""

    def __call__(
        self, end: int, sample_shape: torch.Size = torch.Size()
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return samples from m(x_{0:end}) along with corresponding log potentials"""
        log_potential, (loc, _) = (particle_set := self.init(sample_shape))

        xs = [loc]
        log_potentials = [particle_set.logits]

        for t in range(1, end + 1):
            log_potential, (loc, _) = (particle_set := self.step(t, particle_set))
            xs.append(loc)
            log_potentials.append(log_potential)

        return torch.stack(xs, dim=len(log_potentials[0].shape)), torch.stack(
            log_potentials, dim=-1
        )


class BootstrapFeynmanKac(FeynmanKac):
    def __init__(self, ssm: StateSpaceModel, y_obs: torch.Tensor):
        self.ssm = ssm
        self.y_obs = y_obs  # time, event_shape

    @property
    def batch_shape(self):
        return torch.Size([])

    def __len__(self):
        return self.y_obs.size(0)

    def init(self, sample_shape: torch.Size) -> ParticleSet:
        with ppl.observed(y_0=self.y_obs[0].expand(sample_shape + self.y_obs[0].shape)):
            with ppl.trace() as guide_trace:
                self.ssm.init(sample_shape)

            guide_trace = ppl.prune_trace(guide_trace, "y_0")

            with ppl.replay(guide_trace):
                with ppl.trace() as model_trace:
                    (x, _), stats = self.ssm.init(sample_shape)

        log_m = ppl.trace_log_prob(guide_trace, sample_ndims=len(sample_shape))
        log_joint = ppl.trace_log_prob(model_trace, sample_ndims=len(sample_shape))
        log_potential = log_joint - log_m

        return ParticleSet(log_potential, (x, stats))

    def step(self, t: int, particle_set: ParticleSet) -> ParticleSet:
        assert isinstance(
            particle_set, ParticleSet
        ), f"expected particle_set to be type ParticleSet, got {type(particle_set)}"
        sample_shape = particle_set.logits.shape
        observed_kwargs = {
            f"y_{t}": self.y_obs[t].expand(sample_shape + self.y_obs[t].shape)
        }
        _, (x, stats) = particle_set
        with ppl.observed(**observed_kwargs):
            with ppl.trace() as guide_trace:
                self.ssm.step(t, stats)

            guide_trace = ppl.prune_trace(guide_trace, f"y_{t}")

            with ppl.replay(guide_trace):
                with ppl.trace() as model_trace:
                    (x, _), stats = self.ssm.step(t, stats)

        log_m = ppl.trace_log_prob(guide_trace, sample_ndims=len(sample_shape))
        log_joint = ppl.trace_log_prob(model_trace, sample_ndims=len(sample_shape))
        log_potential = log_joint - log_m

        return ParticleSet(log_potential, (x, stats))


class GuidedFeynmanKac(FeynmanKac):
    def __init__(
        self, guide, ssm: StateSpaceModel, y_obs: torch.Tensor, batch_shape: torch.Size
    ):
        self.guide = guide
        self.ssm = ssm
        self.y_obs = y_obs  # batch_shape, time, event_shape
        self._batch_shape = batch_shape

    @property
    def batch_shape(self):
        return self._batch_shape

    def __len__(self):
        return self.y_obs.size(len(self.batch_shape))

    def init(self, sample_shape: torch.Size) -> ParticleSet:
        y_0 = torch.select(self.y_obs, len(self.batch_shape), 0)
        y_0 = y_0.expand(sample_shape + y_0.shape)
        with ppl.observed(y_0=y_0):
            with ppl.trace() as guide_trace:
                self.guide.init(sample_shape)

            guide_trace = ppl.prune_trace(guide_trace, "y_0")

            with ppl.replay(guide_trace):
                with ppl.trace() as model_trace:
                    (x, _), stats = self.ssm.init(sample_shape + self.guide.batch_shape)

        log_m = ppl.trace_log_prob(
            guide_trace, sample_ndims=len(sample_shape) + len(self.batch_shape)
        )
        log_joint = ppl.trace_log_prob(
            model_trace, sample_ndims=len(sample_shape) + len(self.batch_shape)
        )
        log_potential = log_joint - log_m  # pyright: ignore

        return ParticleSet(log_potential, (x, stats))

    def step(self, t: int, particle_set: ParticleSet) -> ParticleSet:
        sample_shape = particle_set.logits.shape[:1]
        y_t = torch.select(self.y_obs, len(self.batch_shape), t)
        y_t = y_t.expand(sample_shape + y_t.shape)
        observed_kwargs = {f"y_{t}": y_t}
        _, (x, stats) = particle_set
        with ppl.observed(**observed_kwargs):
            with ppl.trace() as guide_trace:
                self.guide.step(t, stats, sample_shape)

            guide_trace = ppl.prune_trace(guide_trace, f"y_{t}")
            assert f"x_{t}" in guide_trace

            with ppl.replay(guide_trace):
                with ppl.trace() as model_trace:
                    (x, _), stats = self.ssm.step(t, stats)

        log_m = ppl.trace_log_prob(
            guide_trace, sample_ndims=len(sample_shape) + len(self.batch_shape)
        )
        log_joint = ppl.trace_log_prob(
            model_trace, sample_ndims=len(sample_shape) + len(self.batch_shape)
        )
        log_potential = log_joint - log_m  # pyright: ignore

        return ParticleSet(log_potential, (x, stats))
