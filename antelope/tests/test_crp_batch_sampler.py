import itertools
import torch

from antelope.samplers import CRPBatchSampler


def test_crp_batch_sampler_sample():
    # generate some dummy data
    class_datasets = [torch.randn(50, 3) for _ in range(100)]
    sampler = CRPBatchSampler(class_datasets, 1.0, 10, 20)
    X, y = sampler.sample()
    assert X.size() == (10, 20, 3)
    assert y.size() == (10, 20)
    assert y.dtype == torch.int64


def test_crp_batch_sampler_iter():
    class_datasets = [torch.randn(50, 3) for _ in range(100)]
    sampler = CRPBatchSampler(class_datasets, 1.0, 10, 20)
    i = 0
    for X, y in itertools.islice(sampler, 5):
        assert X.size() == (10, 20, 3)
        assert y.size() == (10, 20)
        i += 1
    assert i == 5
