from functools import partial
from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as jr

from jax import vmap

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class CanonicalLabeling:
    def __init__(self):
        self.num_labels = 0
        self.mapping = {}
        self.inverse_mapping = {}

    def transform(self, labels):
        assert isinstance(labels, list), "labels must be a list of labels"
        res = []
        for label in labels:
            if label not in self.mapping:
                self.mapping[label] = self.num_labels
                self.inverse_mapping[self.num_labels] = label
                self.num_labels += 1
            res.append(self.mapping[label])
        return res

    def inverse_transform(self, canonical_labels):
        assert isinstance(
            canonical_labels, list
        ), "canonical labels must be a list of labels"
        return [self.inverse_mapping[label] for label in canonical_labels]


def _nonparametric_pad_1d(arr: jax.Array, value):
    assert arr.ndim == 1
    ind = (arr == 0.0).argmax(-1)
    return arr.at[ind].set(value)


def _nonparametric_pad(arr: jax.Array, value):
    if arr.ndim == 1:
        return _nonparametric_pad_1d(arr, value)
    elif arr.ndim == 2:
        return vmap(_nonparametric_pad_1d, in_axes=(0, None))(arr, value)
    else:
        raise ValueError(f"unexpected ndim {arr.ndim}")


def _crp_sample(
    key: jr.PRNGKeyArray, num_timesteps: int, alpha: float, init_counts: jax.Array
):
    def f(counts, key):
        weights = _nonparametric_pad(counts, alpha)
        assignment = tfd.Multinomial(
            1, probs=weights / jnp.sum(weights, axis=-1, keepdims=True)
        ).sample(seed=key)
        return counts + assignment, assignment.argmax(-1)

    _, assignments = lax.scan(f, init_counts, jr.split(key, num_timesteps))

    return assignments.T


def time_reduce(x: jax.Array, time_reduction: str):
    if time_reduction == "none":
        return x
    elif time_reduction == "sum":
        return jnp.sum(x, axis=-1)
    elif time_reduction == "mean":
        return jnp.mean(x, axis=-1)
    else:
        raise ValueError(f"unexpected time_reduction {time_reduction}")


class CRPSampler:
    def __init__(self, alpha: float, max_clusters: int = 1000, safe_mode: bool = True):
        self.alpha = alpha
        self.max_clusters = 1000 if max_clusters is None else max_clusters
        self.safe_mode = safe_mode

    def sample(self, key: jr.PRNGKeyArray, shape: Tuple[int]):
        num_timesteps = shape[-1]
        if len(shape) == 1:
            init_counts = jnp.zeros(min(self.max_clusters, num_timesteps))
        elif len(shape) == 2:
            batch_size = shape[0]
            init_counts = jnp.zeros((batch_size, min(self.max_clusters, num_timesteps)))
        else:
            raise ValueError(f"unsupported shape {shape}")
        return _crp_sample(key, num_timesteps, self.alpha, init_counts)

    def conditional_sample(self, key: jr.PRNGKeyArray, shape: Tuple, init: jax.Array):
        init_counts = jnp.sum(nn.one_hot(init, self.max_clusters), axis=0)
        num_timesteps = shape[-1]
        base_sample = partial(
            _crp_sample,
            num_timesteps=num_timesteps,
            alpha=self.alpha,
            init_counts=init_counts,
        )
        if len(shape) == 1:
            return base_sample(key)
        elif len(shape) == 2:
            batch_size = shape[0]
            return vmap(base_sample, in_axes=0)(jr.split(key, batch_size))
        else:
            raise ValueError(f"unsupported shape {shape}")

    def logits(self, y: jax.Array):
        if self.safe_mode:
            assert y.ndim == 1
            assert jnp.max(y) < self.max_clusters
        y_onehot = nn.one_hot(y, self.max_clusters)
        counts = jnp.pad(jnp.cumsum(y_onehot, 0), ((1, 0), (0, 0)), mode="constant")[
            :-1
        ]
        weights = _nonparametric_pad(counts, self.alpha)
        return jnp.log(weights)

    def conditional_logits(self, y_init: jax.Array, y_rest: jax.Array):
        if self.safe_mode:
            assert y.ndim == 1
            assert jnp.max(y_init) < self.max_clusters
            assert jnp.max(y_rest) < self.max_clusters
        counts_init = jnp.sum(nn.one_hot(y_init, self.max_clusters), 0)

        y_onehot = nn.one_hot(y_rest, self.max_clusters)
        counts = (
            jnp.pad(jnp.cumsum(y_onehot, 0), ((1, 0), (0, 0)), mode="constant")[:-1]
            + counts_init
        )
        weights = _nonparametric_pad(counts, self.alpha)
        return jnp.log(weights)

    def predictive_log_proba(self, y):
        return nn.log_softmax(self.logits(y), axis=-1)

    def log_likelihood(self, y: jax.Array, time_reduction: str = "sum"):
        logits = self.logits(y)
        log_proba = nn.log_softmax(logits, axis=-1)
        log_like = log_proba.at[jnp.arange(log_proba.shape[0]), y].get()
        return time_reduce(log_like, time_reduction)

    def nats_per_timestep(self, y: jax.Array):
        return -self.log_likelihood(y, time_reduction="mean")

    def perplexity(self, y: jax.Array):
        return jnp.exp(self.nats_per_timestep(y))

    def conditional_log_likelihood(
        self, y_init: jax.Array, y_rest: jax.Array, time_reduction: str = "sum"
    ):
        log_like = self.log_likelihood(
            jnp.concatenate([y_init, y_rest], axis=-1), time_reduction="none"
        )[..., y_init.shape[-1] :]

        return time_reduce(log_like, time_reduction)

    def conditional_nats_per_timestep(self, y_init: jax.Array, y_rest: jax.Array):
        return -self.conditional_log_likelihood(y_init, y_rest, time_reduction="mean")

    def conditional_perplexity(self, y_init: jax.Array, y_rest: jax.Array):
        return jnp.exp(self.conditional_nats_per_timestep(y_init, y_rest))
