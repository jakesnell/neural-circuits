import pytest

import torch
from torch.distributions import Normal, Independent
from torch.testing import assert_close

from partikel import ppl


def test_basic_trace():
    def model():
        return ppl.sample("x", Normal(0, 1))

    with ppl.trace() as capture:
        x = model()

    assert_close(x, capture["x"]["value"])


def test_double_trace():
    def model():
        return ppl.sample("x", Normal(0, 1))

    # check raises
    with pytest.raises(AssertionError):
        with ppl.trace():
            model()
            model()

    # ensure that previous trace properly restores context after exception
    with ppl.trace() as capture:
        x = model()

    assert_close(x, capture["x"]["value"])


def test_log_prob():
    base_dist = Normal(0, 1)

    with ppl.trace() as capture:
        x = ppl.sample("x", base_dist)
        y = ppl.sample("y", base_dist)

    log_p = ppl.trace_log_prob(capture)
    assert log_p.shape == torch.Size([])
    assert_close(log_p, base_dist.log_prob(x) + base_dist.log_prob(y))


def test_log_prob_univariate_shape():
    base_dist = Normal(0, 1)
    n = 10

    def sample_closure(name):
        return ppl.sample(name, base_dist, sample_shape=torch.Size([n]))

    with ppl.trace() as capture:
        x = sample_closure("x")
        y = sample_closure("y")

    # test default behavior (sum over all dimensions)
    log_p = ppl.trace_log_prob(capture)
    assert log_p.shape == torch.Size([])
    assert_close(log_p, base_dist.log_prob(x).sum() + base_dist.log_prob(y).sum())

    # test custom ndims
    log_p_sample = ppl.trace_log_prob(capture, sample_ndims=1)
    assert log_p_sample.shape == torch.Size([n])
    assert_close(log_p_sample, base_dist.log_prob(x) + base_dist.log_prob(y))


def test_log_prob_bivariate_shape():
    base_dist = Independent(Normal(0, 1).expand(torch.Size([2])), 1)
    n = 10
    m = 3

    def sample_closure(name):
        return ppl.sample(name, base_dist, sample_shape=torch.Size([m, n]))

    with ppl.trace() as capture:
        x = sample_closure("x")
        y = sample_closure("y")

    # test default behavior (sum over all dimensions)
    log_p = ppl.trace_log_prob(capture)
    assert log_p.shape == torch.Size([])
    assert_close(log_p, base_dist.log_prob(x).sum() + base_dist.log_prob(y).sum())

    # test ndim = 1
    log_p_sample = ppl.trace_log_prob(capture, sample_ndims=1)
    assert log_p_sample.shape == torch.Size([m])
    assert_close(
        log_p_sample, base_dist.log_prob(x).sum(-1) + base_dist.log_prob(y).sum(-1)
    )

    # test ndim = 2
    log_p_sample = ppl.trace_log_prob(capture, sample_ndims=2)
    assert log_p_sample.shape == torch.Size([m, n])
    assert_close(log_p_sample, base_dist.log_prob(x) + base_dist.log_prob(y))


def test_observed():
    def model():
        x = ppl.sample("x", Normal(0, 1))
        y = ppl.sample("y", Normal(x, 0.2))
        return x, y

    y_obs = torch.ones(())

    with ppl.observed(y=y_obs):
        _, y = model()

    assert_close(y, y_obs)


def test_observed_log_joint():
    prior = Normal(0, 1)
    likelihood = lambda x: Normal(x, 0.2)

    def model():
        x = ppl.sample("x", prior)
        y = ppl.sample("y", likelihood(x))
        return x, y

    y_obs = torch.ones(())

    with ppl.observed(y=y_obs):
        with ppl.trace() as capture:
            x, _ = model()

    assert_close(
        ppl.trace_log_prob(capture), prior.log_prob(x) + likelihood(x).log_prob(y_obs)
    )


def test_log_potential():
    prior = Normal(0, 1)
    likelihood = lambda x: Normal(x, 0.2)
    proposal = lambda y: Normal(y, 1)

    def model():
        x = ppl.sample("x", prior)
        y = ppl.sample("y", likelihood(x))
        return x, y

    def guide(y):
        return ppl.sample("x", proposal(y))

    y_obs = torch.ones(())

    with ppl.trace() as guide_trace:
        x_proposal = guide(y_obs)

    with ppl.observed(y=y_obs):
        with ppl.replay(guide_trace):
            with ppl.trace() as model_trace:
                model()

    assert_close(
        ppl.trace_log_prob(model_trace) - ppl.trace_log_prob(guide_trace),
        prior.log_prob(x_proposal)
        + likelihood(x_proposal).log_prob(y_obs)
        - proposal(y_obs).log_prob(x_proposal),
    )

    def implicit_guide():
        y = ppl.observe("y")
        return ppl.sample("x", proposal(y))

    with ppl.observed(y=y_obs):
        with ppl.trace() as guide_trace:
            x_proposal = implicit_guide()

        with ppl.replay(guide_trace):
            with ppl.trace() as model_trace:
                model()

    assert_close(
        ppl.trace_log_prob(model_trace)
        - ppl.trace_log_prob(ppl.prune_observed(guide_trace)),
        prior.log_prob(x_proposal)
        + likelihood(x_proposal).log_prob(y_obs)
        - proposal(y_obs).log_prob(x_proposal),
    )


def test_prune():
    prior = Normal(0, 1)

    def model():
        x = ppl.sample("x", prior)
        y = ppl.sample("y", prior)
        return x, y

    # check ordinary case
    with ppl.trace() as model_trace:
        x, y = model()

    assert_close(ppl.trace_log_prob(model_trace), prior.log_prob(x) + prior.log_prob(y))
    assert_close(
        ppl.trace_log_prob(ppl.prune_trace(model_trace, ["y"])), prior.log_prob(x)
    )


def test_prune_observed():
    prior = Normal(0, 1)

    def model():
        x = ppl.sample("x", prior)
        y = ppl.observe("y")
        return x, y

    # check ordinary case
    with ppl.observed(y=torch.ones(())):
        with ppl.trace() as model_trace:
            x, y = model()

    pruned_trace = ppl.prune_observed(model_trace)
    assert "x" in pruned_trace
    assert_close(pruned_trace["x"]["value"], x)
    assert "y" not in pruned_trace
