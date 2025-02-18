import pytest
import math

import torch
from torch.distributions import Normal, Independent
from torch.distributions.multivariate_normal import MultivariateNormal

from partikel.transition import LinearGaussianTransition
from partikel.emission import GaussianEmission
from partikel.state_space_model import (
    HomogeneousStateSpaceModel,
    CollapsedInfiniteMixtureModel,
)
from partikel.feynman_kac import BootstrapFeynmanKac, GuidedFeynmanKac
from partikel.exponential_family import (
    NaturalNormalInverseGammaPrior,
    NormalInverseGamma,
    NaturalNormalInverseGammaLikelihood,
    NaturalGaussianPrior,
    NaturalGaussianLikelihood,
)
from partikel.markov_process import ChineseRestaurantProcess
from partikel.guide import CollapsedInfiniteMixtureGuide


@pytest.fixture
def t():
    return 20


@pytest.fixture
def num_samples():
    return 30


@pytest.fixture
def scalar_fk(t):
    rho = 0.9
    sigma_x = 1.0
    sigma_y = 0.2

    ssm = HomogeneousStateSpaceModel(
        Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))),
        LinearGaussianTransition(rho, sigma_x),
        GaussianEmission(sigma_y),
    )
    (_, y), _ = ssm(t)
    return BootstrapFeynmanKac(ssm, y)


def test_scalar_unit_sample_shape(scalar_fk, t):
    x, log_potential = scalar_fk(t)
    assert x.shape == torch.Size([t + 1])
    assert log_potential.shape == torch.Size([t + 1])


def test_scalar_general_sample_shape(scalar_fk, t, num_samples):
    x, log_potential = scalar_fk(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert log_potential.shape == torch.Size([num_samples, t + 1])


@pytest.fixture
def bivariate_fk(t):
    rho = 0.9
    sigma_x = 1.0
    sigma_y = 0.2

    ssm = HomogeneousStateSpaceModel(
        Independent(
            Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))).expand(torch.Size([2])),
            1,
        ),
        LinearGaussianTransition(rho, sigma_x),
        lambda stats: MultivariateNormal(
            stats.loc, covariance_matrix=sigma_y**2 * torch.eye(2)
        ),
    )
    (_, y), _ = ssm(t)
    return BootstrapFeynmanKac(ssm, y)


def test_bivariate_unit_sample_shape(bivariate_fk, t):
    x, log_g = bivariate_fk(t)
    assert x.shape == torch.Size([t + 1, 2])
    assert log_g.shape == torch.Size([t + 1])


def test_bivariate_general_sample_shape(bivariate_fk, t, num_samples):
    x, log_g = bivariate_fk(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1, 2])
    assert log_g.shape == torch.Size([num_samples, t + 1])


def test_guided_fk():
    ssm = CollapsedInfiniteMixtureModel(
        ChineseRestaurantProcess(1.0),
        NaturalNormalInverseGammaPrior.from_ordinary(
            NormalInverseGamma(
                loc=torch.zeros(2),
                mean_concentration=5 * torch.ones(2),
                concentration=torch.ones(2),
                scale=torch.ones(2),
            )
        ),
        NaturalNormalInverseGammaLikelihood(),
    )
    (x, y), _ = ssm(199, torch.Size([3]))
    assert x.shape == torch.Size([3, 200])
    assert y.shape == torch.Size([3, 200, 2])

    guide = CollapsedInfiniteMixtureGuide(ssm, y, torch.Size([3]))
    fk = GuidedFeynmanKac(guide, ssm, y, torch.Size([3]))

    x, log_g = fk(199, sample_shape=torch.Size([5]))
    assert x.shape == torch.Size([5, 3, 200])
    assert log_g.shape == torch.Size([5, 3, 200])


# TODO: make base and guide ssm different
# base_ssm = CollapsedInfiniteMixtureModel(
#     ChineseRestaurantProcess(1.0),
#     NaturalGaussianPrior.from_ordinary(
#         Normal(torch.zeros(2), torch.ones(2)),
#     ),
#     NaturalGaussianLikelihood(0.2),
# )
