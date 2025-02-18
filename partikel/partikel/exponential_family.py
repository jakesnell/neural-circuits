import abc
import math
from typing import Callable
from typing_extensions import Self

from attrs import define, field, validators
import itertools

import torch
from torch.distributions import (
    Distribution,
    ExpTransform,
    Normal,
    LogNormal,
    StudentT,
    Beta,
    Bernoulli,
    TransformedDistribution,
)
from pyro.distributions import InverseGamma, MaskedDistribution
from torch.distributions import constraints
from torch.testing import assert_close
from torch_scatter import scatter_add

from partikel.util import sum_structure

## Auxiliary Distributions


class ChainedDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, init_dist, closure, log_prob_closure=None):
        self.init_dist = init_dist
        self.closure = closure
        self.log_prob_closure = log_prob_closure

    @property
    def batch_shape(self):
        return self.init_dist.batch_shape

    @property
    def event_shape(self):
        return self.init_dist.event_shape

    def sample(self, sample_shape: torch.Size):
        init_samples = self.init_dist.sample(sample_shape)
        return self.closure(init_samples).sample()

    def log_prob(self, x: torch.Tensor):
        if self.log_prob_closure is None:
            raise NotImplementedError
        else:
            return self.log_prob_closure(x)


class MaskedDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, mask_dist, base_dist, log_prob_closure=None):
        self.mask_dist = mask_dist
        self.base_dist = base_dist
        self.log_prob_closure = log_prob_closure

    @property
    def batch_shape(self):
        return self.mask_dist.batch_shape

    @property
    def event_shape(self):
        return self.base_dist.event_shape

    def sample(self, sample_shape: torch.Size):
        data = self.base_dist.sample(sample_shape)
        mask = self.mask_dist.sample(sample_shape)
        return mask * data

    def log_prob(self, x: torch.Tensor):
        if self.log_prob_closure is None:
            raise NotImplementedError
        else:
            return self.log_prob_closure(x)


class NormalInverseGamma(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "mean_concentration": constraints.positive,
        "concentration": constraints.positive,
        "scale": constraints.positive,
    }

    @property
    def _batch_shape(self):
        return self.loc.shape

    @property
    def _event_shape(self):
        return torch.Size()

    def __init__(self, loc, mean_concentration, concentration, scale):
        self.loc = loc
        self.mean_concentration = mean_concentration
        self.concentration = concentration
        self.scale = scale

    def sample(self, sample_shape: torch.Size):
        var = InverseGamma(concentration=self.concentration, rate=self.scale).sample(
            sample_shape
        )
        loc = Normal(self.loc, torch.sqrt(var / self.mean_concentration)).sample()
        return loc, var


## ABCs


class NaturalLikelihood(abc.ABC):
    def counts(self, x: torch.Tensor):
        """Return the counts for updating nu"""
        return torch.ones_like(x)

    def predictive_log_prob_closure(self, prior) -> Callable:
        def log_prob(x: torch.Tensor) -> torch.Tensor:
            log_normalizer_delta = (
                prior.update(self.suff_stats(x), self.counts(x)).log_normalizer()
                - prior.log_normalizer()
            )
            return self.log_base_measure(x) + log_normalizer_delta

        return log_prob

    @abc.abstractmethod
    def suff_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return the sufficient statistics"""

    @abc.abstractmethod
    def log_base_measure(self, x: torch.Tensor) -> torch.Tensor:
        """Return log h(x) in the natural parameterization"""

    @abc.abstractmethod
    def predictive_distribution(self, prior):
        """Return the predictive distribution by marginalizing over the prior"""


@define
class NaturalConjugatePrior(abc.ABC):
    tau: tuple[torch.Tensor, ...] = field(validator=validators.instance_of(tuple))
    nu: torch.Tensor = field(validator=validators.instance_of(torch.Tensor))

    def params(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return the natural parameters of this distribution"""
        return (self.tau, self.nu)

    def map_params(self, f) -> Self:
        return self.__class__(tuple(map(f, self.tau)), f(self.nu))

    def zip_params(self, other, f) -> Self:
        return self.__class__(
            tuple(itertools.starmap(f, zip(self.tau, other.tau))), f(self.nu, other.nu)
        )

    def update(self, t: tuple[torch.Tensor, ...], n: torch.Tensor) -> Self:
        """Update self with observed sufficient statistics"""
        assert len(t) == len(self.tau)

        def sum_broadcast(a, b):
            a, b = torch.broadcast_tensors(a, b)
            return a + b

        # tau = tuple(tau_i.expand_as(t_i) + t_i for t_i, tau_i in zip(t, self.tau))
        # nu = self.nu.expand_as(n) + n
        tau = tuple(sum_broadcast(tau_i, t_i) for t_i, tau_i in zip(t, self.tau))
        nu = sum_broadcast(self.nu, n)
        return self.__class__(tau, nu)

    def indexed_update(
        self,
        index: torch.Tensor,
        t: tuple[torch.Tensor, ...],
        n: torch.Tensor,
    ) -> Self:
        assert len(t) == len(self.tau)

        def index_as(target):
            return index.view(
                index.shape + torch.Size([1] * (target.ndim - index.ndim))
            ).expand(index.shape + torch.Size([1]) + target.shape[index.ndim + 1 :])

        tau = tuple(
            scatter_add(
                t_i.unsqueeze(index.ndim),
                index_as(tau_i),
                dim=index.ndim,
                out=tau_i.clone(),
            )
            for t_i, tau_i in zip(t, self.tau)
        )
        nu = scatter_add(
            n.unsqueeze(index.ndim),
            index_as(self.nu),
            dim=index.ndim,
            out=self.nu.clone(),
        )
        return self.__class__(tau, nu)

    def log_predictive_prob(
        self, likelihood: NaturalLikelihood, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log posterior predictive"""
        return (
            likelihood.log_base_measure(x)
            + self.update(
                likelihood.suff_stats(x), likelihood.counts(x)
            ).log_normalizer()
            - self.log_normalizer()
        )

    def log_marginal_likelihood(
        self, likelihood: NaturalLikelihood, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log marginal likelihood probability of x"""
        return torch.sum(
            likelihood.log_base_measure(x).sum(
                dim=tuple(range(x.ndim - self.tau[0].ndim))
            )
            + self.update(
                sum_structure(
                    likelihood.suff_stats(x),
                    dim_closure=lambda v: tuple(range(v.ndim - self.tau[0].ndim)),
                ),  # pyright: ignore
                sum_structure(
                    likelihood.counts(x),
                    dim_closure=lambda v: tuple(range(v.ndim - self.nu.ndim)),
                ),  # pyright: ignore
            ).log_normalizer()
            - self.log_normalizer(),
            tuple(range(self.tau[0].ndim)),
        )

    @abc.abstractclassmethod
    def from_ordinary(cls, dist: Distribution) -> Self:
        """Construct from ordinary distribution"""

    @abc.abstractmethod
    def to_ordinary(self) -> Distribution:
        """Convert from natural parameterization back to ordinary"""

    @abc.abstractmethod
    def log_normalizer(self) -> torch.Tensor:
        """Return B(\tau, \nu) in the definition of the exponential family conjugate prior"""


## Likelihoods


class NaturalHurdleLikelihood(NaturalLikelihood):
    def __init__(self, base_likelihood: NaturalLikelihood):
        self.base_likelihood = base_likelihood

    def suff_stats(self, x: torch.Tensor):
        mask = x.ne(0.0)
        base_suff_stats = tuple(
            torch.where(mask, t, 0.0) for t in self.base_likelihood.suff_stats(x)
        )
        return (mask.type(x.dtype), *base_suff_stats)

    def log_base_measure(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0.0)
        return torch.where(mask, self.base_likelihood.log_base_measure(x), 0.0)

    # def predictive_log_prob_closure(self, prior) -> Callable:
    #     base_log_prob_closure = self.base_likelihood.predictive_log_prob_closure(
    #         prior.base_dist
    #     )
    #     hurdle_log_prob_closure = (
    #         NaturalBernoulliLikelihood().predictive_log_prob_closure(prior.hurdle_dist)
    #     )

    #     def log_prob(x: torch.Tensor) -> torch.Tensor:
    #         mask = x.ne(0.0)

    #         return hurdle_log_prob_closure(mask.type(x.dtype)) + torch.where(
    #             mask, torch.nan_to_num(base_log_prob_closure(x), nan=-float("inf")), 0.0
    #         )

    #     return log_prob

    def predictive_distribution(self, prior) -> Distribution:
        mask_distribution = NaturalBernoulliLikelihood().predictive_distribution(
            prior.hurdle_dist
        )
        base_distribution = self.base_likelihood.predictive_distribution(
            prior.base_dist
        )
        return MaskedDistribution(
            mask_distribution,
            base_distribution,
            log_prob_closure=self.predictive_log_prob_closure(prior),
        )


class NaturalBernoulliLikelihood(NaturalLikelihood):
    def suff_stats(self, x: torch.Tensor):
        return (x,)

    def log_base_measure(self, x: torch.Tensor):
        return torch.zeros_like(x)

    def predictive_log_prob_closure(self, prior) -> Callable:
        alpha, beta = prior.compute_alpha_beta()

        def log_prob(x: torch.Tensor) -> torch.Tensor:
            return torch.log(torch.where(x != 0.0, alpha, beta)) - torch.log(
                alpha + beta
            )

        return log_prob

    def predictive_distribution(self, prior) -> Distribution:
        assert isinstance(prior, NaturalBetaPrior)
        alpha, beta = prior.compute_alpha_beta()
        return Bernoulli(probs=alpha / (alpha + beta))


class NaturalGaussianLikelihood(NaturalLikelihood):
    def __init__(self, scale: float):
        self.scale = scale

    def suff_stats(self, x: torch.Tensor):
        return (x / self.scale**2, -torch.ones_like(x) / (2 * self.scale**2))

    def log_base_measure(self, x: torch.Tensor):
        return (
            -(x**2) / (2 * self.scale**2)
            - 0.5 * math.log(2 * math.pi)
            - math.log(self.scale)
        )

    def predictive_distribution(self, prior) -> Distribution:
        assert isinstance(prior, NaturalGaussianPrior)
        posterior = prior.to_ordinary()
        return Normal(
            loc=posterior.loc,
            scale=torch.sqrt(posterior.scale**2 + self.scale**2),
        )


class NaturalNormalInverseGammaLikelihood(NaturalLikelihood):
    def suff_stats(self, x: torch.Tensor):
        return (x, x**2)

    def log_base_measure(self, x: torch.Tensor):
        return -0.5 * torch.ones_like(x) * math.log(2 * math.pi)

    def predictive_distribution(self, prior) -> StudentT:
        assert isinstance(prior, NaturalNormalInverseGammaPrior)
        posterior = prior.to_ordinary()
        # TODO: verify this
        return StudentT(
            df=2 * posterior.concentration,
            loc=posterior.loc,
            scale=torch.sqrt(
                posterior.scale
                / posterior.concentration
                * (1.0 + 1.0 / posterior.mean_concentration)
            ),
        )


class NaturalFourParameterNormalInverseGammaLikelihood(NaturalLikelihood):
    def suff_stats(self, x: torch.Tensor):
        return (x, -0.5 * (x**2), torch.full_like(x, -0.5))

    def log_base_measure(self, x: torch.Tensor):
        return -0.5 * torch.ones_like(x) * math.log(2 * math.pi)

    def predictive_distribution(self, prior) -> StudentT:
        assert isinstance(prior, NaturalFourParameterNormalInverseGammaPrior)
        posterior = prior.to_ordinary()

        safe_scale = torch.clamp(
            posterior.scale, min=torch.finfo(posterior.scale.dtype).eps
        )

        return StudentT(
            df=2 * posterior.concentration,
            loc=posterior.loc,
            scale=torch.sqrt(
                safe_scale
                / posterior.concentration
                * (1.0 + 1.0 / posterior.mean_concentration)
            ),
        )


class NaturalLogNormalInverseGammaLikelihood(NaturalLikelihood):
    def suff_stats(self, x: torch.Tensor):
        return (torch.log(x), torch.log(x) ** 2)

    def log_base_measure(self, x: torch.Tensor):
        return -torch.log(x) - 0.5 * math.log(2 * math.pi)

    def predictive_log_prob_closure(self, prior) -> Callable:
        return self.predictive_distribution(prior).log_prob

    def predictive_distribution(self, prior):
        assert isinstance(prior, NaturalNormalInverseGammaPrior)
        original_distribution = (
            NaturalNormalInverseGammaLikelihood().predictive_distribution(prior)
        )
        return TransformedDistribution(
            original_distribution, ExpTransform(), validate_args=False
        )


class NaturalFourParameterLogNormalInverseGammaLikelihood(NaturalLikelihood):
    def suff_stats(self, x: torch.Tensor):
        return (torch.log(x), -0.5 * (torch.log(x) ** 2), torch.full_like(x, -0.5))

    def log_base_measure(self, x: torch.Tensor):
        return -torch.log(x) - 0.5 * math.log(2 * math.pi)

    def predictive_distribution(self, prior):
        assert isinstance(prior, NaturalFourParameterNormalInverseGammaPrior)
        original_distribution = (
            NaturalFourParameterNormalInverseGammaLikelihood().predictive_distribution(
                prior
            )
        )
        return TransformedDistribution(
            original_distribution, ExpTransform(), validate_args=False
        )


## Conjugate Priors


class NaturalBetaPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, dist: Beta) -> Self:
        alpha = dist.concentration1
        beta = dist.concentration0
        tau = (alpha - 1,)
        nu = alpha + beta - 2
        return cls(tau, nu)

    def compute_alpha_beta(self):
        (tau,) = self.tau
        alpha = tau + 1
        beta = self.nu - tau + 1
        return alpha, beta

    def to_ordinary(self) -> Beta:
        alpha, beta = self.compute_alpha_beta()
        return Beta(concentration1=alpha, concentration0=beta)

    def log_normalizer(self) -> torch.Tensor:
        alpha, beta = self.compute_alpha_beta()
        return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)


class NaturalGaussianPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, dist: Normal) -> Self:
        mu = dist.loc
        sigma = dist.scale
        tau = (mu / (sigma**2), -1.0 / (2 * sigma**2))
        nu = torch.zeros_like(mu)
        return cls(tau, nu)

    def to_ordinary(self) -> Normal:
        tau_0, tau_1 = self.tau
        sigma = torch.sqrt(-1.0 / (2 * tau_1))
        mu = -tau_0 / (2 * tau_1)
        return Normal(loc=mu, scale=sigma)

    def log_normalizer(self) -> torch.Tensor:
        tau_0, tau_1 = self.tau
        return (
            -(tau_0**2) / (4 * tau_1)
            - 0.5 * torch.log(-2 * tau_1)
            + 0.5 * math.log(2 * math.pi)
        )


class NaturalNormalInverseGammaPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, dist: NormalInverseGamma) -> Self:
        assert_close(dist.mean_concentration, 2 * dist.concentration + 3.0)
        nu = dist.mean_concentration
        tau_0 = nu * dist.loc
        tau_1 = nu * (dist.loc**2) + 2.0 * dist.scale
        return cls((tau_0, tau_1), nu)

    def to_ordinary(self) -> NormalInverseGamma:
        tau_0, tau_1 = self.tau
        loc = tau_0 / self.nu
        mean_concentration = self.nu
        concentration = (self.nu - 3.0) / 2.0
        scale = 0.5 * (tau_1 - (tau_0**2) / self.nu)
        return NormalInverseGamma(
            loc=loc,
            mean_concentration=mean_concentration,
            concentration=concentration,
            scale=scale,
        )

    def log_normalizer(self) -> torch.Tensor:
        """Return B(\tau, \nu) in the definition of the exponential family conjugate prior"""
        tau_0, tau_1 = self.tau
        return (
            -0.5 * torch.log(self.nu)
            + 0.5 * math.log(2 * math.pi)
            - 0.5 * (self.nu - 3) * torch.log(0.5 * (tau_1 - (tau_0**2) / self.nu))
            + torch.lgamma(0.5 * (self.nu - 3.0))
        )


class NaturalHurdleNormalInverseGammaPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, hurdle_dist: Beta, base_dist: NormalInverseGamma) -> Self:
        natural_hurdle_dist = NaturalBetaPrior.from_ordinary(hurdle_dist)
        natural_base_dist = NaturalNormalInverseGammaPrior.from_ordinary(base_dist)
        assert_close(natural_base_dist.nu, natural_hurdle_dist.tau[0])
        tau = (natural_hurdle_dist.tau[0], *natural_base_dist.tau)
        nu = natural_hurdle_dist.nu
        return cls(tau, nu)

    @property
    def hurdle_dist(self) -> NaturalBetaPrior:
        beta_tau = self.tau[:1]
        return NaturalBetaPrior(beta_tau, self.nu)

    @property
    def base_dist(self) -> NaturalNormalInverseGammaPrior:
        base_tau = self.tau[1:]
        base_nu = self.tau[0]
        return NaturalNormalInverseGammaPrior(base_tau, base_nu)

    def to_ordinary(self) -> tuple[Beta, NormalInverseGamma]:
        return self.hurdle_dist.to_ordinary(), self.base_dist.to_ordinary()

    def log_normalizer(self) -> torch.Tensor:
        """Return B(\tau, \nu) in the definition of the exponential family conjugate prior"""
        return self.hurdle_dist.log_normalizer() + self.base_dist.log_normalizer()


class NaturalFourParameterNormalInverseGammaPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, dist: NormalInverseGamma) -> Self:
        nu = 2 * dist.concentration + 3.0
        tau_0 = dist.mean_concentration * dist.loc
        tau_1 = -dist.scale - 0.5 * dist.mean_concentration * (dist.loc**2)
        tau_2 = -0.5 * dist.mean_concentration
        return cls((tau_0, tau_1, tau_2), nu)

    def to_ordinary(self) -> NormalInverseGamma:
        tau_0, tau_1, tau_2 = self.tau
        loc = -tau_0 / (2 * tau_2)
        mean_concentration = -2.0 * tau_2
        concentration = (self.nu - 3.0) / 2.0
        scale = -tau_1 + (tau_0**2) / (4.0 * tau_2)

        return NormalInverseGamma(
            loc=loc,
            mean_concentration=mean_concentration,
            concentration=concentration,
            scale=scale,
        )

    def log_normalizer(self) -> torch.Tensor:
        """Return B(\tau, \nu) in the definition of the exponential family conjugate prior"""
        ordinary = self.to_ordinary()
        return (
            -0.5 * torch.log(ordinary.mean_concentration)
            - ordinary.concentration * torch.log(ordinary.scale)
            + torch.lgamma(ordinary.concentration)
            + 0.5 * math.log(2 * math.pi)
        )


class NaturalHurdleFourParameterNormalInverseGammaPrior(NaturalConjugatePrior):
    @classmethod
    def from_ordinary(cls, hurdle_dist: Beta, base_dist: NormalInverseGamma) -> Self:
        natural_hurdle_dist = NaturalBetaPrior.from_ordinary(hurdle_dist)
        natural_base_dist = NaturalFourParameterNormalInverseGammaPrior.from_ordinary(
            base_dist
        )
        assert_close(natural_base_dist.nu, natural_hurdle_dist.tau[0])
        tau = (natural_hurdle_dist.tau[0], *natural_base_dist.tau)
        nu = natural_hurdle_dist.nu
        return cls(tau, nu)

    @property
    def hurdle_dist(self) -> NaturalBetaPrior:
        beta_tau = self.tau[:1]
        return NaturalBetaPrior(beta_tau, self.nu)

    @property
    def base_dist(self) -> NaturalFourParameterNormalInverseGammaPrior:
        base_tau = self.tau[1:]
        base_nu = self.tau[0]
        return NaturalFourParameterNormalInverseGammaPrior(base_tau, base_nu)

    def to_ordinary(self) -> tuple[Beta, NormalInverseGamma]:
        return self.hurdle_dist.to_ordinary(), self.base_dist.to_ordinary()

    def log_normalizer(self) -> torch.Tensor:
        """Return B(\tau, \nu) in the definition of the exponential family conjugate prior"""
        return self.hurdle_dist.log_normalizer() + self.base_dist.log_normalizer()
