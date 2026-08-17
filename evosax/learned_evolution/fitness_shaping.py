"""Fitness shaping utilities for backward compatibility with learned evolution."""

import jax
import jax.numpy as jnp

from evosax.types import Fitness, Solution


def standardize(values: jax.Array) -> jax.Array:
    """Standardize globally using the learned checkpoints' feature contract."""
    mean = jnp.nanmean(values)
    std = jnp.nanstd(values)
    return (values - mean) / (std + 1e-10)


def normalize(arr: jax.Array, min_val: float = -1.0, max_val: float = 1.0) -> jax.Array:
    """Normalize values into the requested feature range."""
    arr = jnp.clip(arr, -1e10, 1e10)
    return (max_val - min_val) * (arr - jnp.nanmin(arr)) / (
        jnp.nanmax(arr) - jnp.nanmin(arr) + 1e-10
    ) + min_val


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
