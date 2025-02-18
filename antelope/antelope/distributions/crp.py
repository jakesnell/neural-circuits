from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import jax.random as jr


class CRP:
    def __init__(self, alpha: float, max_clusters: int):
        self.alpha = float(alpha)
        self.max_clusters = int(max_clusters)

    def sample(self, shape, *, seed: jr.PRNGKeyArray):
        assert len(shape) == 1
        return jax.vmap(lambda seed: self.sample_n(1, seed=seed))(
            jr.split(seed, shape[0])
        )

    def sample_n(self, n: int, *, seed: jr.PRNGKeyArray):
        class SampleState(NamedTuple):
            k: jax.Array
            counts: jax.Array

        new_cluster_logit = jnp.log(self.alpha).reshape(1)

        def init_state():
            return SampleState(
                k=jnp.zeros((), dtype=jnp.int32),
                counts=jnp.zeros((self.max_clusters,), dtype=jnp.int32),
            )

        def f(carry, key):
            (k_prev, counts_prev) = carry
            z = jr.categorical(
                key, jnp.concatenate([new_cluster_logit, jnp.log(counts_prev)], -1)
            )

            k, z = lax.cond(
                z > 0, lambda: (k_prev, z - 1), lambda: (k_prev + 1, k_prev)
            )

            return SampleState(k=k, counts=counts_prev.at[z].add(1)), z

        _, samples = lax.scan(f, init_state(), jr.split(seed, n))

        return samples
