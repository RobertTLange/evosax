import chex
import jax.numpy as jnp
from typing import Tuple


def full_eigen_decomp(
    C: chex.Array, B: chex.Array, D: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Perform eigendecomposition of covariance matrix."""
    if B is not None and D is not None:
        return C, B, D

    # Symmetry
    C = (C + C.T) / 2

    # Diagonal loading
    eps = 1e-8
    C = C + eps * jnp.eye(C.shape[0])

    # Compute eigendecomposition
    D2, B = jnp.linalg.eigh(C)

    # Sort eigenvalues and eigenvectors
    idx = jnp.argsort(D2, descending=True)
    D2 = D2[idx]
    B = B[:, idx]

    # More conservative thresholding
    D = jnp.sqrt(jnp.maximum(D2, eps))
    return C, B, D


def diag_eigen_decomp(C: chex.Array, D: chex.Array) -> chex.Array:
    """Perform simplified decomposition of diagonal covariance matrix."""
    if D is not None:
        return D

    eps = 1e-8
    D = jnp.sqrt(jnp.maximum(C, eps))
    return D
