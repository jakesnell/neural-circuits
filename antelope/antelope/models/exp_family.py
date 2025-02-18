import torch
import torch.nn as nn
from torch.distributions import Beta

from partikel.guide import CollapsedInfiniteMixtureGuide
from partikel.markov_process import ChineseRestaurantProcess
from partikel.state_space_model import CollapsedInfiniteMixtureModel
from partikel.exponential_family import (
    NormalInverseGamma,
    NaturalHurdleNormalInverseGammaPrior,
    NaturalHurdleLikelihood,
    NaturalLogNormalInverseGammaLikelihood,
    NaturalNormalInverseGammaLikelihood,
    NaturalHurdleFourParameterNormalInverseGammaPrior,
    NaturalFourParameterLogNormalInverseGammaLikelihood,
    NaturalFourParameterNormalInverseGammaPrior,
    NaturalFourParameterNormalInverseGammaLikelihood,
)


class ExponentialFamilyParticleFilter(nn.Module):
    def __init__(self, in_dim: int, alpha: float):
        super().__init__()
        self.in_dim = in_dim
        self.alpha = alpha
        self.loc = nn.Parameter(torch.zeros(in_dim))
        self.log_conc = nn.Parameter(torch.zeros(in_dim))
        self.log_scale = nn.Parameter(torch.zeros(in_dim))
        self.sparse_prior_logit = nn.Parameter(torch.zeros(in_dim))

    @property
    def conc(self):
        return self.log_conc.exp()

    @property
    def scale(self):
        return self.log_scale.exp()

    @property
    def prior_sparsity(self):
        return torch.sigmoid(self.sparse_prior_logit)

    def construct_ssm(self):
        nig_ordinary = NormalInverseGamma(
            loc=self.loc,
            mean_concentration=(2 * self.conc + 3),
            concentration=self.conc,
            scale=self.scale,
        )
        hurdle = Beta(
            concentration1=nig_ordinary.mean_concentration + 1,
            concentration0=(nig_ordinary.mean_concentration + 1)
            / torch.exp(-self.sparse_prior_logit),
        )

        conjugate_prior = NaturalHurdleNormalInverseGammaPrior.from_ordinary(
            hurdle, nig_ordinary
        )
        likelihood = NaturalHurdleLikelihood(NaturalLogNormalInverseGammaLikelihood())

        ssm = CollapsedInfiniteMixtureModel(
            ChineseRestaurantProcess(self.alpha, cuda=self.loc.is_cuda),
            conjugate_prior,
            likelihood,
        )
        return ssm

    def construct_guide(self, ssm, X, batch_shape=None):
        if batch_shape is None:
            batch_shape = torch.Size([X.shape[0]])

        return CollapsedInfiniteMixtureGuide(
            ssm,
            X,
            batch_shape,
        )

    def nll_trace(self, X, z):
        ssm = self.construct_ssm()
        guide = self.construct_guide(ssm, X)
        return -guide.trace_log_prob(z.long())

    def loss(self, X, z):
        return self.nll_trace(X, z).mean()


class FourParameterExponentialFamilyParticleFilter(nn.Module):
    def __init__(self, in_dim: int, alpha: float, eps: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.alpha = alpha
        self.loc = nn.Parameter(torch.zeros(in_dim))
        self.log_mean_conc = nn.Parameter(torch.zeros(in_dim))
        self.log_conc = nn.Parameter(torch.zeros(in_dim))
        self.log_scale = nn.Parameter(torch.zeros(in_dim))
        self.sparse_prior_logit = nn.Parameter(torch.zeros(in_dim))
        self.eps = eps

    @property
    def mean_conc(self):
        return torch.clamp(self.log_mean_conc, max=12.0).exp()

    @property
    def conc(self):
        return self.log_conc.exp()

    @property
    def scale(self):
        return self.log_scale.exp()

    @property
    def prior_sparsity(self):
        return torch.sigmoid(self.sparse_prior_logit)

    def construct_ssm(self):
        nig_ordinary = NormalInverseGamma(
            loc=self.loc,
            mean_concentration=self.mean_conc,
            concentration=self.conc,
            scale=self.scale,
        )
        nu = 2 * self.conc + 3
        hurdle = Beta(
            concentration1=nu + 1,
            concentration0=(nu + 1) / torch.exp(-self.sparse_prior_logit),
        )

        conjugate_prior = (
            NaturalHurdleFourParameterNormalInverseGammaPrior.from_ordinary(
                hurdle, nig_ordinary
            )
        )
        likelihood = NaturalHurdleLikelihood(
            NaturalFourParameterLogNormalInverseGammaLikelihood()
        )

        ssm = CollapsedInfiniteMixtureModel(
            ChineseRestaurantProcess(self.alpha, cuda=self.loc.is_cuda),
            conjugate_prior,
            likelihood,
        )
        return ssm

    def construct_guide(self, ssm, X, batch_shape=None):
        if batch_shape is None:
            batch_shape = torch.Size([X.shape[0]])

        return CollapsedInfiniteMixtureGuide(
            ssm,
            X,
            batch_shape,
        )

    def nll_trace(self, X, z):
        ssm = self.construct_ssm()
        guide = self.construct_guide(ssm, X)
        return -guide.trace_log_prob(z.long())

    def loss(self, X, z):
        return self.nll_trace(X, z).mean()


class FourParameterNoHurdleExponentialFamilyParticleFilter(nn.Module):
    def __init__(self, in_dim: int, alpha: float):
        super().__init__()
        self.in_dim = in_dim
        self.alpha = alpha
        self.loc = nn.Parameter(torch.zeros(in_dim))
        self.log_mean_conc = nn.Parameter(torch.zeros(in_dim))
        self.log_conc = nn.Parameter(torch.zeros(in_dim))
        self.log_scale = nn.Parameter(torch.zeros(in_dim))

    @property
    def mean_conc(self):
        return torch.clamp(self.log_mean_conc, max=12.0).exp()

    @property
    def conc(self):
        return self.log_conc.exp()

    @property
    def scale(self):
        return self.log_scale.exp()

    def construct_ssm(self):
        nig_ordinary = NormalInverseGamma(
            loc=self.loc,
            mean_concentration=self.mean_conc,
            concentration=self.conc,
            scale=self.scale,
        )

        conjugate_prior = NaturalFourParameterNormalInverseGammaPrior.from_ordinary(
            nig_ordinary
        )
        likelihood = NaturalFourParameterNormalInverseGammaLikelihood()

        ssm = CollapsedInfiniteMixtureModel(
            ChineseRestaurantProcess(self.alpha, cuda=self.loc.is_cuda),
            conjugate_prior,
            likelihood,
        )
        return ssm

    def construct_guide(self, ssm, X, batch_shape=None):
        if batch_shape is None:
            batch_shape = torch.Size([X.shape[0]])

        return CollapsedInfiniteMixtureGuide(
            ssm,
            X,
            batch_shape,
        )

    def nll_trace(self, X, z):
        ssm = self.construct_ssm()
        guide = self.construct_guide(ssm, X)
        return -guide.trace_log_prob(z.long())

    def loss(self, X, z):
        return self.nll_trace(X, z).mean()
