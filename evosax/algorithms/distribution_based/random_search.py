"""Random Search."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
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
    best_solution_shaped: Solution
    best_fitness_shaped: float


@struct.dataclass
class Params(BaseParams):
    pass


class RandomSearch(DistributionBasedAlgorithm):
    """Random Search."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        sampling_fn: Callable,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize Random Search."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)
        self.sampling_fn = sampling_fn

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
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
        keys = jax.random.split(key, self.population_size)
        population = jax.vmap(self.sampling_fn)(keys)
        population = jax.vmap(self._ravel_solution)(population)
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
            best_solution_shaped=best_solution_shaped,
            best_fitness_shaped=best_fitness_shaped,
        )
