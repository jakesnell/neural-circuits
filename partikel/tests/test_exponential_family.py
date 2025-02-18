import contextlib

import torch
from torch.distributions import Beta, Normal, LogNormal
from torch.testing import assert_close

from partikel.exponential_family import (
    NaturalBernoulliLikelihood,
    NaturalBetaPrior,
    NaturalGaussianPrior,
    NaturalGaussianLikelihood,
    NaturalHurdleLikelihood,
    NaturalHurdleNormalInverseGammaPrior,
    NaturalLogNormalInverseGammaLikelihood,
    NaturalNormalInverseGammaPrior,
    NaturalFourParameterLogNormalInverseGammaLikelihood,
    NaturalFourParameterNormalInverseGammaLikelihood,
    NaturalFourParameterNormalInverseGammaPrior,
    NaturalHurdleFourParameterNormalInverseGammaPrior,
)
from partikel.exponential_family import (
    NormalInverseGamma,
)
from partikel.util import sum_structure

import pytest
from pytest import approx


@contextlib.contextmanager
def default_dtype(dtype):
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)

    try:
        yield
    finally:
        torch.set_default_dtype(orig_dtype)


def test_conversion():
    prior = Normal(2.0, 0.1)
    natural = NaturalGaussianPrior.from_ordinary(prior)

    assert len(natural.tau) == 2
    assert natural.tau[0].shape == torch.Size([])
    assert natural.tau[1].shape == torch.Size([])

    assert natural.tau[0].item() == approx(200.0)
    assert natural.tau[1].item() == approx(-50.0)

    converted = natural.to_ordinary()
    assert converted.loc == prior.loc
    assert converted.scale == prior.scale


def test_conversion_1d():
    prior = Normal(torch.tensor([2.0, 1.0]), torch.tensor([0.1, 0.2]))
    natural = NaturalGaussianPrior.from_ordinary(prior)

    assert len(natural.tau) == 2
    assert natural.tau[0].shape == torch.Size([2])
    assert natural.tau[1].shape == torch.Size([2])

    assert natural.tau[0][0].item() == approx(200.0)
    assert natural.tau[1][0].item() == approx(-50.0)
    assert natural.tau[0][1].item() == approx(25.0)
    assert natural.tau[1][1].item() == approx(-12.5)

    converted = natural.to_ordinary()
    assert_close(converted.loc, prior.loc)
    assert_close(converted.scale, prior.scale)


def test_gaussian_likelihood():
    sigma = 0.1
    likelihood = NaturalGaussianLikelihood(sigma)

    n = 50
    x = torch.rand(n)

    t = likelihood.suff_stats(x)
    assert len(t) == 2
    assert_close(t[0], x / (sigma**2))
    assert_close(t[1], -torch.ones_like(x) / (2 * sigma**2))


def test_log_posterior_predictive():
    with default_dtype(torch.float64):
        mu_0 = torch.tensor([2.0, 1.0])
        sigma_0 = torch.tensor([0.1, 0.2])
        sigma = 0.5

        prior = Normal(mu_0, sigma_0)
        natural_prior = NaturalGaussianPrior.from_ordinary(prior)
        natural_likelihood = NaturalGaussianLikelihood(sigma)
        n = 10
        x = torch.randn(n, 2)

        sigma_n = 1 / torch.sqrt(n / (sigma**2) + 1.0 / (sigma_0**2))
        mu_n = (sigma_n**2) * (mu_0 / (sigma_0**2) + torch.sum(x, 0) / (sigma**2))

        y = torch.randn(3, 5, 2)
        ordinary_posterior_predictive = Normal(
            mu_n, torch.sqrt(sigma_n**2 + sigma**2)
        )
        natural_posterior = natural_prior.update(
            sum_structure(natural_likelihood.suff_stats(x), dim=0),
            sum_structure(natural_likelihood.counts(x), dim=0),
        )

        assert_close(
            ordinary_posterior_predictive.log_prob(y),
            natural_posterior.log_predictive_prob(natural_likelihood, y),
        )


def test_marginal_likelihood():
    mu_0 = torch.tensor([2.0, 1.0])
    sigma_0 = torch.tensor([0.1, 0.2])
    sigma = 0.5

    prior = NaturalGaussianPrior.from_ordinary(Normal(mu_0, sigma_0))
    likelihood = NaturalGaussianLikelihood(sigma)

    n = 30
    x = torch.randn(n, 2)

    posterior = prior
    lml = torch.zeros(())
    for i in range(x.size(0)):
        lml += posterior.log_predictive_prob(likelihood, x[i]).sum(-1)
        posterior = posterior.update(
            likelihood.suff_stats(x[i]), likelihood.counts(x[i])
        )

    assert_close(lml, prior.log_marginal_likelihood(likelihood, x))


def test_beta():
    alpha = torch.tensor([1.0, 2.0])
    beta = torch.tensor([6.0, 1.1])
    conjugate_prior = NaturalBetaPrior.from_ordinary(Beta(alpha, beta))

    y = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    f_log_prob = NaturalBernoulliLikelihood().predictive_log_prob_closure(
        conjugate_prior
    )

    assert_close(
        f_log_prob(y),
        NaturalBernoulliLikelihood()
        .predictive_distribution(conjugate_prior)
        .log_prob(y),
    )


def test_hurdle():
    conc = torch.rand(2)
    mean_conc = 2 * conc + 3.0
    prior_sparsity = torch.rand(2)

    hurdle = Beta(
        concentration1=mean_conc + 1,
        concentration0=(mean_conc + 1) * (prior_sparsity / (1 - prior_sparsity)),
    )
    conjugate_prior = NaturalHurdleNormalInverseGammaPrior.from_ordinary(
        hurdle,
        NormalInverseGamma(torch.randn(2), mean_conc, conc, torch.rand(2)),
    )

    likelihood = NaturalHurdleLikelihood(NaturalLogNormalInverseGammaLikelihood())

    f_log_prob = likelihood.predictive_log_prob_closure(conjugate_prior)

    y = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, torch.rand(1).item()],
            [torch.rand(1).item(), 0.0],
            [torch.rand(1).item(), torch.rand(1).item()],
        ]
    )

    super_f_log_prob = super(
        likelihood.__class__, likelihood
    ).predictive_log_prob_closure(conjugate_prior)

    assert_close(
        f_log_prob(y), likelihood.predictive_distribution(conjugate_prior).log_prob(y)
    )
    assert_close(
        super_f_log_prob(y),
        likelihood.predictive_distribution(conjugate_prior).log_prob(y),
    )

    assert_close(super_f_log_prob(y), f_log_prob(y))


def test_normal_inverse_gamma():
    conc = torch.rand(2)
    mean_conc = 2 * conc + 3.0

    conjugate_prior = NaturalNormalInverseGammaPrior.from_ordinary(
        NormalInverseGamma(torch.randn(2), mean_conc, conc, torch.rand(2)),
    )

    likelihood = NaturalLogNormalInverseGammaLikelihood()

    f_log_prob = likelihood.predictive_log_prob_closure(conjugate_prior)

    y = torch.rand(10, 2)

    super_f_log_prob = super(
        likelihood.__class__, likelihood
    ).predictive_log_prob_closure(conjugate_prior)

    assert_close(
        f_log_prob(y), likelihood.predictive_distribution(conjugate_prior).log_prob(y)
    )
    assert_close(
        super_f_log_prob(y),
        likelihood.predictive_distribution(conjugate_prior).log_prob(y),
    )

    assert_close(super_f_log_prob(y), f_log_prob(y))


def test_four_parameter_normal_inverse_gamma():
    conc = torch.rand(2)
    mean_conc = torch.full(torch.Size([2]), 3.0)

    ordinary_prior = NormalInverseGamma(torch.randn(2), mean_conc, conc, torch.rand(2))

    # Ordinary prior should not work due to coupling of mean_conc and conc
    with pytest.raises(AssertionError):
        conjugate_prior = NaturalNormalInverseGammaPrior.from_ordinary(ordinary_prior)

    conjugate_prior = NaturalFourParameterNormalInverseGammaPrior.from_ordinary(
        ordinary_prior
    )

    # Test normal
    y = torch.randn(10, 2)

    likelihood = NaturalFourParameterNormalInverseGammaLikelihood()
    f_log_prob = likelihood.predictive_log_prob_closure(conjugate_prior)

    super_f_log_prob = super(
        likelihood.__class__, likelihood
    ).predictive_log_prob_closure(conjugate_prior)

    assert_close(
        f_log_prob(y), likelihood.predictive_distribution(conjugate_prior).log_prob(y)
    )
    assert_close(
        super_f_log_prob(y),
        likelihood.predictive_distribution(conjugate_prior).log_prob(y),
    )

    assert_close(super_f_log_prob(y), f_log_prob(y))

    # Test lognormal
    y = torch.rand(10, 2)

    likelihood = NaturalFourParameterLogNormalInverseGammaLikelihood()
    f_log_prob = likelihood.predictive_log_prob_closure(conjugate_prior)

    super_f_log_prob = super(
        likelihood.__class__, likelihood
    ).predictive_log_prob_closure(conjugate_prior)

    assert_close(
        f_log_prob(y), likelihood.predictive_distribution(conjugate_prior).log_prob(y)
    )
    assert_close(
        super_f_log_prob(y),
        likelihood.predictive_distribution(conjugate_prior).log_prob(y),
    )

    assert_close(super_f_log_prob(y), f_log_prob(y))

    # Test hurdle + lognormal
    y = torch.rand(10, 2)
    y[y < 0.5] = 0.0

    prior_sparsity = torch.rand(2)
    nu = 2 * conc + 3

    hurdle = Beta(
        concentration1=nu + 1,
        concentration0=(nu + 1) * (prior_sparsity / (1 - prior_sparsity)),
    )
    conjugate_prior = NaturalHurdleFourParameterNormalInverseGammaPrior.from_ordinary(
        hurdle,
        NormalInverseGamma(torch.randn(2), mean_conc, conc, torch.rand(2)),
    )

    likelihood = NaturalHurdleLikelihood(
        NaturalFourParameterLogNormalInverseGammaLikelihood()
    )
    f_log_prob = likelihood.predictive_log_prob_closure(conjugate_prior)

    super_f_log_prob = super(
        likelihood.__class__, likelihood
    ).predictive_log_prob_closure(conjugate_prior)

    assert_close(
        f_log_prob(y), likelihood.predictive_distribution(conjugate_prior).log_prob(y)
    )
    assert_close(
        super_f_log_prob(y),
        likelihood.predictive_distribution(conjugate_prior).log_prob(y),
    )
