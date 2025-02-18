import abc
from typing import Callable

from attrs import define
import itertools

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Independent
import tree
from torch_scatter import scatter_add

from partikel import ppl
from partikel.markov_process import MarkovProcess
from partikel.stats import MarkovStats, Stats
from partikel.exponential_family import NaturalConjugatePrior, NaturalLikelihood


class StateSpaceModel(abc.ABC):
    @abc.abstractmethod
    def init(
        self, sample_shape: torch.Size
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Stats]:
        """Return (x_0, y_0) and the initial stats"""

    @abc.abstractmethod
    def step(
        self, t: int, stats: Stats
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Stats]:
        """Return (x_t, y_t) given stats"""

    def __call__(
        self, end: int, sample_shape: torch.Size = torch.Size()
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Stats]:
        (x, y), stats = self.init(sample_shape=sample_shape)

        xs = [x]
        ys = [y]

        for t in range(1, end + 1):
            (x, y), stats = self.step(t, stats)
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs, dim=len(sample_shape))
        ys = torch.stack(ys, dim=len(sample_shape))
        return (xs, ys), stats


class HomogeneousStateSpaceModel(StateSpaceModel):
    def __init__(
        self,
        initial: Distribution,
        transition_closure: Callable,
        emission_closure: Callable,
    ):
        self.initial = initial
        self.transition_closure = transition_closure
        self.emission_closure = emission_closure

    def init(
        self, sample_shape: torch.Size
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Stats]:
        x_0 = ppl.sample("x_0", self.initial, sample_shape=sample_shape)
        stats = MarkovStats(x_0)
        y_0 = ppl.sample("y_0", self.emission_closure(stats))
        return (x_0, y_0), stats

    def step(
        self, t: int, stats: Stats
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Stats]:
        x_t = ppl.sample(f"x_{t}", self.transition_closure(stats))
        stats = MarkovStats(x_t)
        y_t = ppl.sample(f"y_{t}", self.emission_closure(stats))
        return (x_t, y_t), stats


@define
class IMMStats:
    cluster_inds: torch.Tensor
    cluster_params: torch.Tensor
    k: torch.Tensor
    prior_stats: Stats


class InfiniteMixtureModel(StateSpaceModel):
    def __init__(
        self,
        prior: MarkovProcess,
        base_distribution: Distribution,
        emission_closure: Callable[[torch.Tensor], Distribution],
    ):
        self.prior = prior
        self.base_distribution = base_distribution
        self.emission_closure = emission_closure

    def init(
        self, sample_shape: torch.Size
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], IMMStats]:
        x_0, prior_stats = self.prior.init(sample_shape)
        stats = IMMStats(
            cluster_inds=x_0,
            cluster_params=self.base_distribution.sample(x_0.shape[:1]).unsqueeze(1),
            k=torch.ones_like(x_0),
            prior_stats=prior_stats,
        )
        sampled_cluster_params = stats.cluster_params[
            torch.arange(stats.cluster_inds.shape[0]), stats.cluster_inds
        ]
        y_0 = ppl.sample("y_0", self.emission_closure(sampled_cluster_params))
        return (x_0, y_0), stats

    def step(
        self, t: int, stats: IMMStats
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], IMMStats]:
        x_t, prior_stats = self.prior.step(t, stats.prior_stats)

        # extend cluster params if necessary
        if torch.any(x_t >= stats.cluster_params.size(1)):
            cluster_params = torch.zeros(
                torch.Size(
                    [stats.cluster_params.shape[0], 2 * stats.cluster_params.shape[1]]
                    + list(stats.cluster_params.shape[2:])
                ),
                dtype=stats.cluster_params.dtype,
            )
            cluster_params[:, : stats.cluster_params.shape[1]] = stats.cluster_params
        else:
            cluster_params = stats.cluster_params.clone()

        # find the indices where we create a new cluster
        base_inds = torch.nonzero(x_t >= stats.k).view(-1)
        cluster_params[base_inds, x_t[base_inds]] = self.base_distribution.sample(
            sample_shape=torch.Size([len(base_inds)])
        )
        k = stats.k.clone()
        k[base_inds] += 1

        sampled_cluster_params = cluster_params[torch.arange(x_t.shape[0]), x_t]
        y_t = ppl.sample(f"y_{t}", self.emission_closure(sampled_cluster_params))

        stats = IMMStats(
            cluster_inds=x_t,
            cluster_params=cluster_params,
            k=k,
            prior_stats=prior_stats,
        )
        return (x_t, y_t), stats


@define
class CollapsedIMMStats:
    prior_stats: Stats
    posterior: NaturalConjugatePrior


class CollapsedInfiniteMixtureModel(StateSpaceModel):
    def __init__(
        self,
        prior: MarkovProcess,
        conjugate_prior: NaturalConjugatePrior,
        likelihood: NaturalLikelihood,
        max_clusters: int = 50,
    ):
        self.prior = prior
        self.conjugate_prior = conjugate_prior
        self.likelihood = likelihood
        self.max_clusters = max_clusters

    def init(
        self, sample_shape: torch.Size
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], CollapsedIMMStats]:
        # x_0 is the cluster index and has shape = sample_shape
        x_0, prior_stats = self.prior.init(sample_shape)

        # y_0 is the sampled observation and has shape = sample_shape + [D]
        y_0 = ppl.sample(
            "y_0",
            self.likelihood.predictive_distribution(self.conjugate_prior),
            sample_shape=sample_shape,
        )

        posterior = self.conjugate_prior.map_params(
            lambda t: t.expand(
                sample_shape + torch.Size([self.max_clusters]) + t.shape
            ).contiguous()
        ).indexed_update(
            x_0, self.likelihood.suff_stats(y_0), self.likelihood.counts(y_0)
        )

        stats = CollapsedIMMStats(prior_stats, posterior=posterior)

        return (x_0, y_0), stats

    def step(
        self, t: int, stats: CollapsedIMMStats
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], CollapsedIMMStats]:
        x_t, prior_stats = self.prior.step(t, stats.prior_stats)

        assert torch.all(x_t < stats.posterior.tau[0].size(x_t.ndim))
        posterior = stats.posterior

        def select_cluster(t: torch.Tensor):
            x_t_flat = x_t.view(-1)
            t_flat = t.view(x_t_flat.size(0), -1, *t.shape[x_t.ndim + 1 :])
            indexed = t_flat[torch.arange(x_t_flat.size(0)), x_t_flat]
            return indexed.view(*x_t.shape, *t.shape[x_t.ndim + 1 :])

        # gather correct params for sampling
        cluster_predictive = self.likelihood.predictive_distribution(
            posterior.map_params(select_cluster)
        )

        cluster_predictive = Independent(
            cluster_predictive,
            reinterpreted_batch_ndims=len(cluster_predictive.batch_shape) - x_t.ndim,
        )
        y_t = ppl.sample(f"y_{t}", cluster_predictive)

        updated_posterior = posterior.indexed_update(
            x_t, self.likelihood.suff_stats(y_t), self.likelihood.counts(y_t)
        )

        stats = CollapsedIMMStats(prior_stats, updated_posterior)
        return (x_t, y_t), stats
