"""Base module for population-based algorithms.

Population-based algorithms maintain an explicit population of solutions and
evolve it with operators such as selection, crossover and mutation.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from ..base import (
    EvolutionaryAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    """State shared by all population-based algorithms.

    Attributes:
        population: Flat population of shape `(population_size, num_dims)`.
        fitness: Shaped fitness of `population`.
    """

    population: Population
    fitness: Fitness  # Shaped fitness


@struct.dataclass
class Params(BaseParams):
    """Hyperparameters shared by all population-based algorithms."""

    pass


class PopulationBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for population-based algorithms.

    A population-based algorithm stores a population and its fitness in the
    state. `ask` derives new candidates from the stored population and `tell`
    updates the stored population from their fitness. The state is initialized
    from an already evaluated initial population.

    Args:
        population_size: Number of candidate solutions sampled and evaluated
            per generation.
        solution: Example solution PyTree. Only its structure and leaf shapes
            are used: they set the number of search dimensions and how flat
            search vectors are unravelled into the PyTrees returned by `ask`.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied
            to the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function
            `(key, population, fitness, state, params) -> metrics` evaluated
            in `tell`; its output is returned alongside the new state.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        params: Params,
    ) -> State:
        """Initialize the algorithm state from an evaluated population.

        Args:
            key: PRNG key.
            population: Initial population, a PyTree with the structure of
                `solution` and a leading population axis on every leaf.
            fitness: Raw fitness of the initial population. Lower is better. NaN
                values mark failed evaluations and never become the best
                solution.
            params: Algorithm hyperparameters, e.g. `default_params`.

        Returns:
            The initial algorithm state.
        """
        state = self._init(key, params)

        # Ravel population
        population = jax.vmap(self._ravel_solution)(population)

        # Seed archive from the evaluated initial population (raw fitness).
        # Failed evaluations cannot enter the archive.
        archive_fitness = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
        best_idx = jnp.argmin(archive_fitness)
        best_solution = population[best_idx]
        best_fitness = archive_fitness[best_idx]

        state = state.replace(
            best_solution=best_solution,
            best_fitness=best_fitness,
        )

        # Shape fitness
        raw_fitness = fitness
        fitness = self.fitness_shaping_fn(population, raw_fitness, state, params)
        fitness = self._postprocess_fitness(raw_fitness, fitness)

        state = state.replace(
            population=population,
            fitness=fitness,
        )
        return state

    def get_best_solution(self, state: State) -> Solution:
        """Return the best member of the stored population as a solution PyTree.

        This is the member with the lowest shaped fitness in the current
        population; the best solution seen over the whole run is kept in
        `state.best_solution`.
        """
        best_idx = jnp.argmin(state.fitness)
        solution = self._unravel_solution(state.population[best_idx])
        return solution

    def get_population(self, state: State) -> Population:
        """Return the stored population as a PyTree with a leading population axis."""
        return jax.vmap(self._unravel_solution)(state.population)
