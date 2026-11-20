"""Gaussian Hill Climbing algorithm."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from ..base import update_best_solution_and_fitness
from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    best_solution_shaped: Solution
    best_fitness_shaped: float


@struct.dataclass
class Params(BaseParams):
    pass


class HillClimbing(DistributionBasedAlgorithm):
    """Gaussian Hill Climbing algorithm.

    Samples the population from an isotropic Gaussian around the mean and then
    sets the mean to the best solution found so far. The standard deviation
    follows a fixed schedule.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        std_schedule: Function mapping the generation counter to the standard
            deviation of the search distribution, e.g. an Optax schedule.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759)
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            best_solution_shaped=jnp.full((self.num_dims,), jnp.nan),
            best_fitness_shaped=jnp.inf,
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update best solution and fitness shaped
        best_solution_shaped, best_fitness_shaped = update_best_solution_and_fitness(
            population, fitness, state.best_solution_shaped, state.best_fitness_shaped
        )
        return state.replace(
            mean=best_solution_shaped,
            std=self.std_schedule(state.generation_counter),
            best_solution_shaped=best_solution_shaped,
            best_fitness_shaped=best_fitness_shaped,
        )
