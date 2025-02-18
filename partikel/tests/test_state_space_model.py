import pytest
import math

import torch
from torch.distributions import Normal, Independent, MultivariateNormal

from partikel import ppl
from partikel.transition import LinearGaussianTransition
from partikel.emission import GaussianEmission
from partikel.markov_process import ChineseRestaurantProcess
from partikel.state_space_model import (
    HomogeneousStateSpaceModel,
    InfiniteMixtureModel,
    CollapsedInfiniteMixtureModel,
)
from partikel.exponential_family import (
    NaturalNormalInverseGammaPrior,
    NormalInverseGamma,
    NaturalNormalInverseGammaLikelihood,
    NaturalLogNormalInverseGammaLikelihood,
)


@pytest.fixture
def t():
    return 20


@pytest.fixture
def num_samples():
    return 30


@pytest.fixture
def scalar_ssm():
    rho = 0.9
    sigma_x = 1.0
    sigma_y = 0.2

    return HomogeneousStateSpaceModel(
        Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))),
        LinearGaussianTransition(rho, sigma_x),
        GaussianEmission(sigma_y),
    )


def test_scalar_unit_sample_shape(scalar_ssm, t):
    (x, y), _ = scalar_ssm(t)
    assert x.shape == torch.Size([t + 1])
    assert y.shape == torch.Size([t + 1])


def test_scalar_general_sample_shape(scalar_ssm, t, num_samples):
    (x, y), _ = scalar_ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert y.shape == torch.Size([num_samples, t + 1])


@pytest.fixture
def bivariate_ssm():
    rho = 0.9
    sigma_x = 1.0
    sigma_y = 0.2

    return HomogeneousStateSpaceModel(
        Independent(
            Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))).expand(torch.Size([2])),
            1,
        ),
        LinearGaussianTransition(rho, sigma_x),
        GaussianEmission(sigma_y),
    )


def test_bivariate_unit_sample_shape(bivariate_ssm, t):
    (x, y), _ = bivariate_ssm(t)
    assert x.shape == torch.Size([t + 1, 2])
    assert y.shape == torch.Size([t + 1, 2])


def test_bivariate_unit_sample_shape_trace(bivariate_ssm, t):
    with ppl.trace() as capture:
        bivariate_ssm(t)

    for ind in range(t + 1):
        assert capture[f"x_{ind}"]["value"].shape == torch.Size([2])
        assert capture[f"y_{ind}"]["value"].shape == torch.Size([2])


def test_bivariate_general_sample_shape(bivariate_ssm, t, num_samples):
    (x, y), _ = bivariate_ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1, 2])
    assert y.shape == torch.Size([num_samples, t + 1, 2])


def test_bivariate_general_sample_shape_trace(bivariate_ssm, t, num_samples):
    with ppl.trace() as capture:
        bivariate_ssm(t, sample_shape=torch.Size([num_samples]))

    for ind in range(t + 1):
        assert capture[f"x_{ind}"]["value"].shape == torch.Size([num_samples, 2])
        assert capture[f"y_{ind}"]["value"].shape == torch.Size([num_samples, 2])


def test_imm(t, num_samples):
    alpha = 2.0
    ssm = InfiniteMixtureModel(
        ChineseRestaurantProcess(alpha), Normal(0, 3), lambda mu: Normal(mu, 0.2)
    )
    (x, y), _ = ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert y.shape == torch.Size([num_samples, t + 1])


def test_imm_2d(t, num_samples):
    alpha = 2.0
    ssm = InfiniteMixtureModel(
        ChineseRestaurantProcess(alpha),
        MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)),
        lambda mu: MultivariateNormal(mu, 0.01 * torch.eye(2)),
    )
    (x, y), _ = ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert y.shape == torch.Size([num_samples, t + 1, 2])


def test_collapsed_imm(t, num_samples):
    crp_alpha = 2.0
    conc = 1.5
    ssm = CollapsedInfiniteMixtureModel(
        ChineseRestaurantProcess(crp_alpha),
        NaturalNormalInverseGammaPrior.from_ordinary(
            NormalInverseGamma(
                loc=torch.zeros(2),
                mean_concentration=(2 * conc + 3) * torch.ones(2),
                concentration=conc * torch.ones(2),
                scale=torch.ones(2),
            )
        ),
        NaturalNormalInverseGammaLikelihood(),
    )
    (x, y), _ = ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert y.shape == torch.Size([num_samples, t + 1, 2])


def test_collapsed_log_normal_imm(t, num_samples):
    crp_alpha = 2.0
    conc = 1.5
    ssm = CollapsedInfiniteMixtureModel(
        ChineseRestaurantProcess(crp_alpha),
        NaturalNormalInverseGammaPrior.from_ordinary(
            NormalInverseGamma(
                loc=torch.zeros(2),
                mean_concentration=(2 * conc + 3) * torch.ones(2),
                concentration=conc * torch.ones(2),
                scale=torch.ones(2),
            )
        ),
        NaturalLogNormalInverseGammaLikelihood(),
    )
    (x, y), _ = ssm(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])
    assert y.shape == torch.Size([num_samples, t + 1, 2])
    assert y.gt(0.0).all()
