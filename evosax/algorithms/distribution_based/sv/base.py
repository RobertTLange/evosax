"""Base module for Stein Variational Evolution Strategies."""

from collections.abc import Callable
from functools import partial

import jax
from flax import struct

from evosax.core.kernel import kernel_rbf
from evosax.types import Fitness, Metrics, Population, Solution

from ...base import update_best_solution_and_fitness
from ..base import (
    DistributionBasedAlgorithm,
    Params,
    State,
    identity_fitness_shaping_fn,
    metrics_fn,
)


@struct.dataclass
class State(State):
    mean: Solution


@struct.dataclass
class Params(Params):
    pass


class SV_ES(DistributionBasedAlgorithm):
    """Base class for Stein Variational Evolution Strategy."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize base class for Stein Variational Evolution Strategy."""
        DistributionBasedAlgorithm.__init__(
            self, population_size, solution, fitness_shaping_fn, metrics_fn
        )

        self.num_populations = num_populations
        self.total_population_size = num_populations * population_size

        self.kernel = kernel

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        means: Solution,
        params: Params,
    ) -> State:
        """Initialize distribution-based algorithm."""
        state = self._init(key, params)
        state = state.replace(mean=jax.vmap(self._ravel_solution)(means))
        return state

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        """Ask evolutionary algorithm for new candidate solutions to evaluate."""
        # Generate population
        population, state = self._ask(key, state, params)

        # Reshape population
        population = population.reshape(self.total_population_size, self.num_dims)

        # Unravel population
        population = jax.vmap(self._unravel_solution)(population)

        return population, state

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> tuple[State, Metrics]:
        """Tell evolutionary algorithm fitness for state update."""
        # Ravel population
        population = jax.vmap(jax.vmap(self._ravel_solution))(population)

        # Reshape population and fitness
        population = population.reshape(
            self.num_populations, self.population_size, self.num_dims
        )
        fitness = fitness.reshape(self.num_populations, self.population_size)

        # Update best solution and fitness
        best_solution, best_fitness = jax.vmap(update_best_solution_and_fitness)(
            population, fitness, state.best_solution, state.best_fitness
        )
        state = state.replace(best_solution=best_solution, best_fitness=best_fitness)

        # Compute metrics
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num=self.num_populations)
        metrics = jax.vmap(self.metrics_fn, in_axes=(0, 0, 0, 0, None))(
            keys, population, fitness, state, params
        )

        # Shape fitness
        fitness = jax.vmap(self.fitness_shaping_fn, in_axes=(0, 0, 0, None))(
            population, fitness, state, params
        )

        # Update state
        state = self._tell(key, population, fitness, state, params)
        state = state.replace(generation_counter=state.generation_counter + 1)

        return state, metrics

    def _init(self, key: jax.Array, params: Params) -> State:
        keys = jax.random.split(key, num=self.num_populations)
        state = jax.vmap(super()._init, in_axes=(0, None))(keys, params)
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        keys = jax.random.split(key, num=self.num_populations)
        return jax.vmap(super()._ask, in_axes=(0, 0, None))(keys, state, params)

    def get_mean(self, state: State) -> Solution:
        """Return unravelled mean."""
        mean = jax.vmap(self._unravel_solution)(state.mean)
        return mean
