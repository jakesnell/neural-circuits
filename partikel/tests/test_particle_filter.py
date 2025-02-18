import math
import torch
from torch.distributions import Normal, Independent, MultivariateNormal
from torch.testing import assert_close

from partikel.feynman_kac import BootstrapFeynmanKac
from partikel.transition import LinearGaussianTransition
from partikel.emission import GaussianEmission
from partikel.state_space_model import HomogeneousStateSpaceModel
from partikel.particle_filter import ParticleFilter
from partikel.resampler import MultinomialResampler

from partikel.exponential_family import (
    NaturalNormalInverseGammaPrior,
    NormalInverseGamma,
    NaturalNormalInverseGammaLikelihood,
)
from partikel.markov_process import ChineseRestaurantProcess
from partikel.state_space_model import CollapsedInfiniteMixtureModel
from partikel.guide import CollapsedInfiniteMixtureGuide
from partikel.feynman_kac import GuidedFeynmanKac


def test_scalar():
    sigma_x = 1.0
    sigma_y = 0.2
    rho = 0.9
    t = 20
    num_particles = 50

    ssm = HomogeneousStateSpaceModel(
        Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))),
        LinearGaussianTransition(rho, sigma_x),
        GaussianEmission(sigma_y),
    )
    (_, y), _ = ssm(t)
    fk = BootstrapFeynmanKac(ssm, y)

    result = ParticleFilter(fk, MultinomialResampler()).run(num_particles)
    assert result.particles.shape == torch.Size([num_particles, t + 1])
    assert result.logits.shape == torch.Size([num_particles, t + 1])
    assert_close(result.weights.sum(0), torch.ones(t + 1))


def test_bivariate():
    sigma_x = 1.0
    rho = 0.9
    sigma_y = 0.2
    t = 20
    num_particles = 50

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

    fk = BootstrapFeynmanKac(ssm, y)

    result = ParticleFilter(fk, MultinomialResampler()).run(num_particles)
    assert result.particles.shape == torch.Size([num_particles, t + 1, 2])
    assert result.logits.shape == torch.Size([num_particles, t + 1])
    assert_close(result.weights.sum(0), torch.ones(t + 1))


def test_single_guided_particle_filter():
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

    (x, y), _ = ssm(199, torch.Size([]))
    assert x.shape == torch.Size([200])
    assert y.shape == torch.Size([200, 2])

    guide = CollapsedInfiniteMixtureGuide(ssm, y, torch.Size([]))
    fk = GuidedFeynmanKac(guide, ssm, y_obs=y, batch_shape=torch.Size([]))
    num_particles = 7
    pf = ParticleFilter(fk, MultinomialResampler())
    result = pf.run(num_particles)
    assert result.particles.shape == torch.Size([num_particles, 200])
    assert result.logits.shape == torch.Size([num_particles, 200])


def test_batch_guided_particle_filter():
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

    (x, y), _ = ssm(199, torch.Size([5]))
    assert x.shape == torch.Size([5, 200])
    assert y.shape == torch.Size([5, 200, 2])

    guide = CollapsedInfiniteMixtureGuide(ssm, y, torch.Size([5]))
    fk = GuidedFeynmanKac(guide, ssm, y_obs=y, batch_shape=torch.Size([5]))
    num_particles = 7
    pf = ParticleFilter(fk, MultinomialResampler())
    result = pf.run(num_particles)
    assert result.particles.shape == torch.Size([num_particles, 5, 200])
    assert result.logits.shape == torch.Size([num_particles, 5, 200])
