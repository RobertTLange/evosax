import chex
import jax.numpy as jnp
from typing import Tuple


def full_eigen_decomp(
    C: chex.Array, B: chex.Array, D: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Perform eigendecomposition of covariance matrix."""
    if B is not None and D is not None:
        return C, B, D
    C = (C + C.T) / 2  # Make sure matrix is symmetric
    D2, B = jnp.linalg.eigh(C)
    D = jnp.sqrt(jnp.where(D2 < 0, 1e-20, D2))
    C = jnp.dot(jnp.dot(B, jnp.diag(D ** 2)), B.T)
    return C, B, D


def diag_eigen_decomp(C: chex.Array, D: chex.Array) -> chex.Array:
    """Perform simplified decomposition of diagonal covariance matrix."""
    if D is not None:
        return D
    D = jnp.sqrt(jnp.where(C < 0, 1e-20, C))
    return D
