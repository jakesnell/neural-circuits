import pytest
import math

import torch
from torch.distributions import Normal, Independent  # , Categorical

from partikel.transition import LinearGaussianTransition
from partikel.markov_process import (
    HomogeneousMarkovProcess,
    ChineseRestaurantProcess,
)
from partikel.ppl import trace


@pytest.fixture
def t():
    return 20


@pytest.fixture
def num_samples():
    return 30


@pytest.fixture
def scalar_mp():
    rho = 0.9
    sigma_x = 1.0

    return HomogeneousMarkovProcess(
        Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))),
        LinearGaussianTransition(rho, sigma_x),
    )


def test_scalar_unit_sample_shape(scalar_mp, t):
    x, _ = scalar_mp(t)
    assert x.shape == torch.Size([t + 1])


def test_scalar_unit_sample_shape_trace(scalar_mp, t):
    with trace() as capture:
        scalar_mp(t)

    for ind in range(t + 1):
        key = f"x_{ind}"
        assert key in capture
        assert capture[key]["value"].shape == torch.Size([])


def test_scalar_general_sample_shape(scalar_mp, t, num_samples):
    x, _ = scalar_mp(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])


def test_scalar_general_sample_shape_trace(scalar_mp, t, num_samples):
    with trace() as capture:
        scalar_mp(t, sample_shape=torch.Size([num_samples]))

    for ind in range(t + 1):
        key = f"x_{ind}"
        assert key in capture
        assert capture[key]["value"].shape == torch.Size([num_samples])


@pytest.fixture
def bivariate_mp():
    rho = 0.9
    sigma_x = 1.0

    return HomogeneousMarkovProcess(
        Independent(
            Normal(0, math.sqrt(sigma_x**2 / (1 - rho**2))).expand(torch.Size([2])),
            1,
        ),
        LinearGaussianTransition(rho, sigma_x),
    )


def test_bivariate_unit_sample_shape(bivariate_mp, t):
    x, _ = bivariate_mp(t)
    assert x.shape == torch.Size([t + 1, 2])


def test_bivariate_unit_sample_shape_trace(bivariate_mp, t):
    with trace() as capture:
        bivariate_mp(t)

    for ind in range(t + 1):
        key = f"x_{ind}"
        assert key in capture
        assert capture[key]["value"].shape == torch.Size([2])


def test_bivariate_general_sample_shape(bivariate_mp, t, num_samples):
    x, _ = bivariate_mp(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1, 2])


# class SecondOrderKernel(MarkovKernel):
#     def sample(self, x_prev: torch.Tensor, x_pprev=None):
#         """Return samples from X_cur | X_prev = x_prev"""
#         if x_pprev is None:
#             x_pprev = torch.tensor([0.0])
#         x = Normal(
#             1.2 * torch.rand(x_prev.size(0)) * (x_pprev + x_prev), 0.1**2
#         ).sample()
#         return (x, x_prev)


# def test_higher_order(t, num_samples):
#     mp = HomogeneousMarkovProcess(Normal(0, 1), SecondOrderKernel())
#     x = mp.sample(t, sample_shape=torch.Size([num_samples]))
#     assert x.shape == torch.Size([num_samples, t + 1])


def test_crp(t, num_samples):
    alpha = 2.0
    m = ChineseRestaurantProcess(alpha)
    x, _ = m(t, sample_shape=torch.Size([num_samples]))
    assert x.shape == torch.Size([num_samples, t + 1])


def test_crp_multi_dim(t, num_samples):
    alpha = 2.0
    m = ChineseRestaurantProcess(alpha)
    x, _ = m(t, sample_shape=torch.Size([num_samples, 5]))
    assert x.shape == torch.Size([num_samples, 5, t + 1])


# def test_finite_markov_chain(t, num_samples):
#     prior = Categorical(logits=torch.zeros(2))
#     A = 1 - torch.eye(2)
#     m = HomogeneousMarkovProcess(prior, FiniteMarkovKernel(A))
#     x = m.sample(t - 1, sample_shape=torch.Size([num_samples]))
#     assert x.shape == torch.Size([num_samples, t])
#     assert torch.all(x >= 0)
#     assert torch.all(x <= 1)
#     assert t % 2 == 0
#     assert torch.all(torch.sum(x == 0, -1) == t // 2)
#     assert torch.all(torch.sum(x == 1, -1) == t // 2)
