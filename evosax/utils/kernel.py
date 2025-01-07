from abc import abstractmethod, ABC

import jax.numpy as jnp
from chex import Array


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array, bandwidth: float) -> Array:
        """Compute the kernel function between two input arrays."""
        pass

class RBF(Kernel):
    """Radial Basis Function (RBF) kernel implementation."""
    def __call__(self, x1: Array, x2: Array, bandwidth: float) -> Array:
        return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / bandwidth)
