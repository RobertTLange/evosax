from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: jax.Array, x2: jax.Array, bandwidth: float) -> jax.Array:
        """Compute the kernel function between two input arrays."""
        pass


class RBF(Kernel):
    """Radial Basis Function (RBF) kernel implementation."""

    def __call__(self, x1: jax.Array, x2: jax.Array, bandwidth: float) -> jax.Array:
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / bandwidth)
