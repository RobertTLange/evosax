"""Kernel functions for use in evolutionary algorithms."""

from typing import Protocol

import jax
import jax.numpy as jnp


class RBFKernelParams(Protocol):
    """Parameters required by the RBF kernel."""

    kernel_std: float | jax.Array


def kernel_rbf(x: jax.Array, y: jax.Array, params: RBFKernelParams) -> jax.Array:
    """Radial basis function kernel."""
    dist_sq = jnp.sum(jnp.square((x - y) / params.kernel_std), axis=-1)
    return jnp.exp(-0.5 * dist_sq)
