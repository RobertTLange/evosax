"""Kernel functions for use in evolutionary algorithms.

Kernels have the signature `(x, y, params) -> similarity` and are used by the Stein
Variational Evolution Strategies to couple their search distributions.
"""

from typing import Protocol

import jax
import jax.numpy as jnp


class RBFKernelParams(Protocol):
    """Parameters required by the RBF kernel."""

    kernel_std: float | jax.Array


def kernel_rbf(x: jax.Array, y: jax.Array, params: RBFKernelParams) -> jax.Array:
    """Radial basis function kernel.

    Computes `exp(-0.5 * sum(((x - y) / params.kernel_std) ** 2))`.

    Args:
        x: First point.
        y: Second point.
        params: Hyperparameters with a `kernel_std` field, the kernel bandwidth.

    Returns:
        The kernel value, a scalar for vector inputs.
    """
    dist_sq = jnp.sum(jnp.square((x - y) / params.kernel_std), axis=-1)
    return jnp.exp(-0.5 * dist_sq)
