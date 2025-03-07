"""Fitness shaping utilities for backward compatibility with learned evolution."""

import jax
import jax.numpy as jnp

from evosax.types import Fitness, Solution


def standardize(fitness: jax.Array) -> jax.Array:
    """Return standardized fitness."""
    return jax.nn.standardize(fitness, axis=-1, epsilon=1e-8, where=~jnp.isnan(fitness))


def normalize(arr: jax.Array, min_val: float = -1.0, max_val: float = 1.0) -> jax.Array:
    """Normalize fitness."""
    arr_min = jnp.nanmin(arr)
    arr_max = jnp.nanmax(arr)

    return jnp.where(
        jnp.allclose(arr_max, arr_min),
        jnp.ones_like(arr) * (min_val + max_val) / 2,
        min_val + (max_val - min_val) * (arr - arr_min) / (arr_max - arr_min),
    )


def rank(fitness: Fitness) -> jax.Array:
    """Return ranks between [0, fitness.size - 1] according to fitness."""
    assert fitness.ndim == 1
    idx = jnp.argsort(fitness)
    rank = idx.at[idx].set(jnp.arange(fitness.size))
    return rank


def centered_rank(fitness: Fitness) -> jax.Array:
    """Return centered ranks in [-0.5, 0.5] according to fitness."""
    assert fitness.ndim == 1
    ranks = rank(fitness)
    return ranks / (fitness.size - 1) - 0.5


def l2_norm_sq(solution: Solution) -> jax.Array:
    """Compute squared L2 norm of x_i. Assumes x to have shape (..., num_dims)."""
    return jnp.nanmean(jnp.square(solution), axis=-1)
