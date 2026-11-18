"""Fitness shaping functions for evolutionary algorithms.

This module provides various fitness shaping functions that can be used to transform
raw fitness values. Fitness shaping can help improve the convergence properties of
evolutionary algorithms by normalizing or standardizing fitness values, or by adding
regularization terms like weight decay.

Every fitness shaping function has the signature
`(population, fitness, state, params) -> shaped_fitness`, where `population` is the
flat population of shape `(population_size, num_dims)`, `fitness` the raw fitness of
shape `(population_size,)`, and `state` and `params` the state and hyperparameters of
the algorithm. Pass one as `fitness_shaping_fn` when constructing an algorithm. Lower
shaped fitness is better.
"""

from typing import Protocol

import jax
import jax.numpy as jnp

from evosax.types import Fitness, Params, Population, State


class WeightedFitnessParams(Protocol):
    """Parameters required for weighted fitness shaping."""

    weights: jax.Array


def normalize(
    a: jax.Array, axis: int = -1, minval: float = -1.0, maxval: float = 1.0
) -> jax.Array:
    """Rescale an array linearly to the range `[minval, maxval]`.

    NaN values are ignored when computing the minimum and maximum. If all values
    are equal, the midpoint of the range is returned.

    Args:
        a: Array to normalize.
        axis: Axis along which to normalize.
        minval: Lower bound of the output range.
        maxval: Upper bound of the output range.

    Returns:
        The rescaled array, with the same shape as `a`.
    """
    a_min = jnp.nanmin(a, axis=axis, keepdims=True)
    a_max = jnp.nanmax(a, axis=axis, keepdims=True)

    return jnp.where(
        jnp.isclose(a_max, a_min),
        jnp.ones_like(a) * (minval + maxval) / 2,
        minval + (maxval - minval) * (a - a_min) / (a_max - a_min),
    )


def add_weight_decay(fitness_shaping_fn, weight_decay=0.001):
    """Add weight decay to any fitness shaping function.

    The returned function adds the L2 penalty
    `weight_decay * sum(population**2, axis=-1)` to the raw fitness and then
    applies `fitness_shaping_fn`.

    Args:
        fitness_shaping_fn: Fitness shaping function to wrap.
        weight_decay: Coefficient of the L2 penalty.

    Returns:
        A fitness shaping function with the same signature.
    """

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
    """Return the raw fitness unchanged."""
    return fitness


def standardize_fitness_shaping_fn(
    population: Population, fitness: jax.Array, state: State, params: Params
) -> Fitness:
    """Return fitness standardized to zero mean and unit variance.

    NaN values are excluded from the mean and variance.
    """
    return jax.nn.standardize(fitness, axis=-1, epsilon=1e-8, where=~jnp.isnan(fitness))


def normalize_fitness_shaping_fn(
    population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
    """Return fitness rescaled linearly to `[-1, 1]`.

    NaN values are ignored when computing the range. Constant fitness maps to 0.
    """
    return normalize(fitness, axis=-1)


def _rank_fitness(fitness: Fitness) -> jax.Array:
    rankable_fitness = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
    return jax.scipy.stats.rankdata(rankable_fitness, axis=-1) - 1.0


def centered_rank_fitness_shaping_fn(
    population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
    """Return centered ranks in `[-0.5, 0.5]` according to fitness.

    The member with the lowest fitness receives -0.5 and the one with the highest
    fitness 0.5. NaN values are ranked as the worst.
    """
    ranks = _rank_fitness(fitness)
    return ranks / (fitness.shape[-1] - 1) - 0.5


def weights_fitness_shaping_fn(
    population: Population,
    fitness: Fitness,
    state: State,
    params: WeightedFitnessParams,
) -> Fitness:
    """Return `params.weights` indexed by the fitness rank of each member.

    The member with the lowest fitness receives `params.weights[0]`. NaN values
    are ranked as the worst; if every fitness value is NaN, all weights are zero.
    Requires hyperparameters with a `weights` field, as used by e.g. CMA-ES.
    """
    ranks = _rank_fitness(fitness)
    weights = params.weights[..., ranks.astype(jnp.int32)]
    all_failed = jnp.all(jnp.isnan(fitness), axis=-1, keepdims=True)
    return jnp.where(all_failed, jnp.zeros_like(weights), weights)
