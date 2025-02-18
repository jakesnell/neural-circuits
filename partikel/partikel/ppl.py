import contextlib
from typing import Union
import torch
from torch.distributions import Distribution


def validate_shape(x: torch.Tensor, dist: Distribution, **kwargs):
    if "sample_shape" in kwargs:
        sample_shape = kwargs["sample_shape"]
    else:
        sample_shape = torch.Size([])

    expected_shape = sample_shape + dist.batch_shape + dist.event_shape
    assert (
        x.shape == expected_shape
    ), f"invalidate shape detected! expected {expected_shape}, got {x.shape}"
    return x


def observe(name):
    return sample(name, None)


def sample(name, dist, **kwargs):
    return dist.sample(**kwargs)


def trace_log_prob(
    capture, sample_ndims: int = 0, reduce_structure=True
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    def compute_log_prob(node) -> torch.Tensor:
        log_prob = node["dist"].log_prob(node["value"])
        if sample_ndims == log_prob.ndim:
            return log_prob
        else:
            return log_prob.sum(tuple(range(log_prob.ndim))[sample_ndims:])

    if reduce_structure:
        return sum(map(compute_log_prob, capture.values()))  # pyright: ignore
    else:
        return {k: compute_log_prob(v) for k, v in capture.items()}


def prune_trace(trace, names):
    return {k: v for k, v in trace.items() if k not in names}


def prune_observed(trace):
    return {k: v for k, v in trace.items() if v["dist"] is not None}


@contextlib.contextmanager
def trace(capture=None):
    if capture is None:
        capture = {}

    global sample
    orig_sample = sample

    def traced_sample(name, dist, **kwargs):
        assert name not in capture, f"cannot capture random variable {name} twice!"
        capture[name] = {"value": orig_sample(name, dist, **kwargs), "dist": dist}
        return capture[name]["value"]

    sample = traced_sample

    try:
        yield capture
    finally:
        sample = orig_sample


@contextlib.contextmanager
def replay(capture):
    global sample
    orig_sample = sample

    def replayed_sample(name, dist, **kwargs):
        if name in capture:
            ret = capture[name]["value"]
            return ret if dist is None else validate_shape(ret, dist, **kwargs)
        else:
            return orig_sample(name, dist, **kwargs)

    sample = replayed_sample

    try:
        yield
    finally:
        sample = orig_sample


@contextlib.contextmanager
def observed(**context):
    global sample
    orig_sample = sample

    def observed_sample(name, dist, **kwargs):
        if name in context:
            ret = context[name]
            return ret if dist is None else validate_shape(ret, dist, **kwargs)
        else:
            return orig_sample(name, dist, **kwargs)

    sample = observed_sample

    try:
        yield
    finally:
        sample = orig_sample


@contextlib.contextmanager
def sandbox():
    global sample
    orig_sample = sample

    def sandboxed_sample(name, dist, **kwargs):
        return dist.sample(**kwargs)

    sample = sandboxed_sample

    try:
        yield
    finally:
        sample = orig_sample


@contextlib.contextmanager
def maximum_a_posteriori(**context):
    global sample
    orig_sample = sample

    def maximum_a_posteriori_sample(name, dist, **kwargs):
        ret = dist.mode
        if "sample_shape" in kwargs:
            return ret.expand(kwargs["sample_shape"] + ret.shape)
        else:
            return ret

    sample = maximum_a_posteriori_sample

    try:
        yield
    finally:
        sample = orig_sample
