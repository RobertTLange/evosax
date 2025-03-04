"""Base module for population-based algorithms."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...types import Fitness, Population, Solution
from ..base import EvolutionaryAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    population: Population
    fitness: Fitness


@struct.dataclass
class Params(Params):
    pass


class PopulationBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for population-based algorithms."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize base class for population-based algorithm."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: Params,
    ) -> State:
        """Initialize population-based algorithm."""
        state = self._init(key, params)

        state = state.replace(
            population=jax.vmap(self._ravel_solution)(population),
            fitness=fitness,
        )
        return state

    def get_best_solution(self, state: State) -> Solution:
        """Return unravelled best solution."""
        best_idx = jnp.argmin(state.fitness)
        solution = self._unravel_solution(state.population[best_idx])
        return solution

    def get_population(self, state: State) -> Population:
        """Return unravelled population."""
        return jax.vmap(self._unravel_solution)(state.population)
