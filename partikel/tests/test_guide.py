import torch
from torch.testing import assert_close

from partikel.exponential_family import (
    NaturalNormalInverseGammaPrior,
    NormalInverseGamma,
    NaturalNormalInverseGammaLikelihood,
)
from partikel.markov_process import ChineseRestaurantProcess
from partikel.state_space_model import CollapsedInfiniteMixtureModel
from partikel.guide import CollapsedInfiniteMixtureGuide


def test_trace_log_prob():
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

    (x, y), _ = ssm(199, torch.Size([1]))

    assert x.shape == torch.Size([1, 200])
    assert y.shape == torch.Size([1, 200, 2])

    guide = CollapsedInfiniteMixtureGuide(ssm, y, torch.Size([1]))

    log_prob = guide.trace_log_prob(x)
    assert log_prob.shape == torch.Size([1, 200])
    assert log_prob[:, 0].item() == 0.0

    # test consistency
    log_prob2 = guide.trace_log_prob(x)
    assert log_prob.shape == log_prob2.shape
    assert_close(log_prob, log_prob2)


def test_trace_log_prob_batch():
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
    log_prob = guide.trace_log_prob(x)

    assert log_prob.shape == torch.Size([5, 200])
    assert_close(log_prob[:, 0], torch.zeros(5))

    log_prob2 = guide.trace_log_prob(x)
    assert log_prob.shape == log_prob2.shape
    assert_close(log_prob, log_prob2)
