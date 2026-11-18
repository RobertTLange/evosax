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
    """Random Search.

    Draws every candidate solution independently with `sampling_fn`. The mean
    of the state is the best solution found so far.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        sampling_fn: Function `key -> solution` that draws one candidate solution
            with the structure of `solution`. It is called once per population
            member with its own PRNG key.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        sampling_fn: Callable,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
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
