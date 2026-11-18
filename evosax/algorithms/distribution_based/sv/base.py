"""Base module for Stein Variational Evolution Strategies."""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax

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


class SV_ES(DistributionBasedAlgorithm):
    """Base class for Stein Variational Evolution Strategies.

    Runs `num_populations` search distributions in parallel and couples their
    updates through a kernel between their means. Every state field carries a
    leading axis of size `num_populations`, `ask` returns
    `num_populations * population_size` candidates, and `tell` expects one
    fitness value for each of them.

    Args:
        population_size: Number of candidate solutions sampled per search
            distribution in each generation.
        num_populations: Number of search distributions run in parallel.
        solution: Example solution PyTree. Only its structure and leaf shapes
            are used: they set the number of search dimensions and how flat
            search vectors are unravelled into the PyTrees returned by `ask`.
        kernel: Function `(x, y, params) -> similarity` between the means of
            two search distributions. See `evosax.core.kernel`.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied
            per search distribution to the raw fitness in `tell` before the
            algorithm update. See `evosax.core.fitness_shaping`.
        metrics_fn: Function
            `(key, population, fitness, state, params) -> metrics` evaluated
            per search distribution in `tell`; its output is returned
            alongside the new state.
    """

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
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
        """Initialize the algorithm state from one initial mean per distribution.

        Args:
            key: PRNG key.
            means: Initial means, a PyTree with the structure of `solution` and a
                leading axis of size `num_populations` on every leaf.
            params: Algorithm hyperparameters, e.g. `default_params`.

        Returns:
            The initial algorithm state.
        """
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
        """Ask for a population of candidate solutions to evaluate.

        Args:
            key: PRNG key.
            state: Current algorithm state.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(population, state)`. `population` is a PyTree with the
            structure of `solution` and a leading axis of size
            `num_populations * population_size`.
        """
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
        """Update the algorithm state with the fitness of a population.

        Args:
            key: PRNG key.
            population: Population returned by `ask`.
            fitness: Fitness of each population member, shape
                `(num_populations * population_size,)`. Lower is better.
            state: Algorithm state returned by `ask`.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(state, metrics)` with the updated state and the output of
            `metrics_fn` for each search distribution.
        """
        # Ravel population
        population = jax.vmap(self._ravel_solution)(population)

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

    def _init(self, key: jax.Array, params: Any) -> Any:
        keys = jax.random.split(key, num=self.num_populations)
        state = jax.vmap(super()._init, in_axes=(0, None))(keys, params)
        return state

    def _ask(
        self,
        key: jax.Array,
        state: Any,
        params: Any,
    ) -> tuple[Population, Any]:
        keys = jax.random.split(key, num=self.num_populations)
        return jax.vmap(super()._ask, in_axes=(0, 0, None))(keys, state, params)

    def get_mean(self, state: State) -> Solution:
        """Return the means of all search distributions as a PyTree.

        Every leaf has a leading axis of size `num_populations`.
        """
        mean = jax.vmap(self._unravel_solution)(state.mean)
        return mean
