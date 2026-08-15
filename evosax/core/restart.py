"""Restart utilities for Evolution Strategies."""

from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from evosax.types import Fitness, Params, Population, State

from ..algorithms.distribution_based.cma_es import eigen_decomposition


class GenerationRestartState(Protocol):
    generation_counter: int


class GenerationRestartParams(Protocol):
    generation_threshold: int


class SpreadRestartParams(Protocol):
    fitness_spread_threshold: float


class CMAESRestartState(Protocol):
    C: jax.Array
    mean: jax.Array
    std: float | jax.Array
    p_c: jax.Array


class CMAESRestartParams(Protocol):
    tol_x: float
    tol_x_up: float
    tol_condition_C: float


class AmalgamRestartState(Protocol):
    c_mult: jax.Array


@struct.dataclass
class RestartState:
    restart_counter: int


@struct.dataclass
class RestartParams:
    pass


@struct.dataclass
class FitnessStdRestartParams(RestartParams):
    tol: float = 0.001
    atol: float = 0.0


def generation_cond(
    population: Population,
    fitness: Fitness,
    state: GenerationRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: GenerationRestartParams,
) -> bool:
    """Stop after a certain number of generations."""
    return state.generation_counter >= restart_params.generation_threshold


def spread_cond(
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
    restart_state: RestartState,
    restart_params: SpreadRestartParams,
) -> jax.Array:
    """Stop if fitness max minus fitness min is below threshold."""
    return jnp.max(fitness) - jnp.min(fitness) < restart_params.fitness_spread_threshold


def fitness_std_cond(
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
    restart_state: RestartState,
    restart_params: FitnessStdRestartParams,
) -> bool:
    """Stop if fitness standard deviation is below tolerance."""
    finite = jnp.all(jnp.isfinite(fitness))
    threshold = restart_params.atol + restart_params.tol * jnp.abs(jnp.mean(fitness))
    return jnp.logical_and(finite, jnp.std(fitness) <= threshold)


def cma_cond(
    population: Population,
    fitness: Fitness,
    state: CMAESRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: CMAESRestartParams,
) -> jax.Array:
    """Stop if condition specific to CMA-ES is met.

    Default tolerances:
    tol_x: 1e-12 * sigma
    tol_x_up: 1e4
    tol_condition_C: 1e14
    """
    dC = jnp.diag(state.C)
    _, B, D = eigen_decomposition(state.C)

    # Stop if std of normal distribution is smaller than tolx in all coordinates
    # and pc is smaller than tolx in all components.
    cond_s_1 = jnp.all(state.std * dC < restart_params.tol_x)
    cond_s_2 = jnp.all(state.std * state.p_c < restart_params.tol_x)
    cond_1 = jnp.logical_and(cond_s_1, cond_s_2)

    # Stop if std diverges
    cond_2 = state.std * jnp.max(D) > restart_params.tol_x_up

    # Stop if adding 0.2 std does not change mean.
    cond_no_coord_change = jnp.any(
        state.mean == state.mean + (0.2 * state.std * jnp.sqrt(dC))
    )
    cond_3 = cond_no_coord_change

    # Stop if adding 0.1 std in principal directions of C does not change mean.
    cond_no_axis_change = jnp.all(
        state.mean == state.mean + (0.1 * state.std * D[0] * B[:, 0])
    )
    cond_4 = cond_no_axis_change

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    cond_condition_cov = jnp.max(D) / jnp.min(D) > restart_params.tol_condition_C
    cond_5 = cond_condition_cov

    return cond_1 | cond_2 | cond_3 | cond_4 | cond_5


def amalgam_cond(
    population: Population,
    fitness: Fitness,
    state: AmalgamRestartState,
    params: Params,
    restart_state: RestartState,
    restart_params: RestartParams,
) -> jax.Array:
    """Stop if c_mult is below threshold."""
    return state.c_mult < 1e-10


@struct.dataclass
class IPOPRestartState(RestartState):
    restart_counter: int
    restart_next: bool
    active_population_size: int


@struct.dataclass
class IPOPRestartParams(RestartParams):
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    population_size_multiplier: int = 2
    copy_mean: bool = False


@struct.dataclass
class BIPOPRestartState(RestartState):
    restart_counter: int
    restart_next: bool
    active_population_size: int
    restart_large_counter: int
    large_eval_budget: int
    small_eval_budget: int
    small_pop_active: bool


@struct.dataclass
class BIPOPRestartParams(RestartParams):
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    population_size_multiplier: int = 2
    copy_mean: bool = False
