"""Fitness shaping functions for evolutionary algorithms.

This module provides various fitness shaping functions that can be used to transform
raw fitness values. Fitness shaping can help improve the convergence properties of
evolutionary algorithms by normalizing or standardizing fitness values, or by adding
regularization terms like weight decay.
"""

import jax
import jax.numpy as jnp

from evosax.types import Fitness, Params, Population, State


def normalize(
    a: jax.Array, axis: int = -1, minval: float = -1.0, maxval: float = 1.0
) -> jax.Array:
    """Normalize fitness."""
    a_min = jnp.nanmin(a, axis=axis, keepdims=True)
    a_max = jnp.nanmax(a, axis=axis, keepdims=True)

    return jnp.where(
        jnp.allclose(a_max, a_min),
        jnp.ones_like(a) * (minval + maxval) / 2,
        minval + (maxval - minval) * (a - a_min) / (a_max - a_min),
    )


def add_weight_decay(fitness_shaping_fn, weight_decay=0.001):
    """Add weight decay to any fitness shaping function."""

    def wrapped_fitness_fn(population, fitness, state, params):
        # Add weight decay penalty to raw fitness values
        l2_penalty = weight_decay * jnp.sum(jnp.square(population), axis=-1)
        fitness = fitness + l2_penalty

        # Apply the original fitness shaping function to the penalized fitness
        return fitness_shaping_fn(population, fitness, state, params)

    return wrapped_fitness_fn


def identity_fitness_shaping_fn(
    population: Population, fitness: jax.Array, state: State, params: Params
) -> Fitness:
    """Return fitness."""
    return fitness


def standardize_fitness_shaping_fn(
    population: Population, fitness: jax.Array, state: State, params: Params
) -> Fitness:
    """Return standardized fitness."""
    return jax.nn.standardize(fitness, axis=-1, epsilon=1e-8, where=~jnp.isnan(fitness))


def normalize_fitness_shaping_fn(
    population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
    """Return normalized fitness."""
    return normalize(fitness, axis=-1)


def centered_rank_fitness_shaping_fn(
    population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
    """Return centered ranks in [-0.5, 0.5] according to fitness."""
    ranks = jax.scipy.stats.rankdata(fitness, axis=-1) - 1.0
    return ranks / (fitness.shape[-1] - 1) - 0.5


def weights_fitness_shaping_fn(
    population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
    """Return weights according to fitness."""
    ranks = jax.scipy.stats.rankdata(fitness, axis=-1) - 1.0
    return params.weights[..., ranks.astype(jnp.int32)]
