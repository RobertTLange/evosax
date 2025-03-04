"""Kernel functions for use in evolutionary algorithms."""

import jax
import jax.numpy as jnp

from ..algorithms.base import Params


def kernel_rbf(x: jax.Array, y: jax.Array, params: Params) -> jax.Array:
    """Radial basis function kernel."""
    dist_sq = jnp.sum(jnp.square((x - y) / params.kernel_std), axis=-1)
    return jnp.exp(-0.5 * dist_sq)
