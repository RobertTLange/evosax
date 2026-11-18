"""Base module for evolutionary algorithms.

Defines `EvolutionaryAlgorithm`, the ask-eval-tell interface shared by every
algorithm in evosax, together with the base `State` and `Params` dataclasses.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import (
    Fitness,
    Metrics,
    Params as BaseParams,
    Population,
    Solution,
    State as BaseState,
)


@struct.dataclass
class State(BaseState):
    """State shared by all evolutionary algorithms.

    Attributes:
        best_solution: Flat vector of the best solution evaluated so far.
        best_fitness: Fitness of `best_solution` (lower is better).
        generation_counter: Number of completed `tell` calls.
    """

    best_solution: Solution
    best_fitness: float
    generation_counter: int


@struct.dataclass
class Params(BaseParams):
    """Hyperparameters shared by all evolutionary algorithms."""

    pass


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Compute the default metrics returned by `tell`.

    Args:
        key: PRNG key (unused by the default implementation).
        population: Flat population of shape `(population_size, num_dims)`.
        fitness: Raw fitness of shape `(population_size,)`.
        state: Algorithm state, with the best solution already updated for this
            generation.
        params: Algorithm hyperparameters.

    Returns:
        Dictionary with the generation counter, the best fitness and solution of
        this generation, and the best fitness, solution and solution norm so far.
        The entries for this generation are taken with a plain `argmin` over the
        raw fitness, so they are NaN if the population contains a NaN fitness;
        the entries for the best so far ignore NaN values.
    """
    best_idx_in_generation = jnp.argmin(fitness)
    return {
        "generation_counter": state.generation_counter,
        "best_fitness_in_generation": fitness[best_idx_in_generation],
        "best_solution_in_generation": population[best_idx_in_generation],
        "best_fitness": state.best_fitness,
        "best_solution": state.best_solution,
        "best_solution_norm": jnp.linalg.norm(state.best_solution),
    }


class EvolutionaryAlgorithm:
    """Base class for evolutionary algorithms.

    Every algorithm follows the same ask-eval-tell loop: `init` creates the state,
    `ask` proposes a population of candidate solutions, the user evaluates them,
    and `tell` updates the state from the resulting fitness values. `init`, `ask`
    and `tell` are pure, jit-compiled functions of an explicit `state` and
    `params`, so they compose with `jax.vmap` and `jax.lax.scan`.

    Fitness is minimized. To maximize an objective, negate it before calling
    `tell`.

    Subclasses implement `_init`, `_ask`, `_tell` and `_default_params`.

    Args:
        population_size: Number of candidate solutions sampled and evaluated
            per generation.
        solution: Example solution PyTree. Only its structure and leaf shapes
            are used: they set the number of search dimensions and how flat
            search vectors are unravelled into the PyTrees returned by `ask`.
            The one exception are the restart controllers in
            `evosax.core.restart`, which restart from the values of `solution`
            unless `copy_mean=True`.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied
            to the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function
            `(key, population, fitness, state, params) -> metrics` evaluated
            in `tell`; its output is returned alongside the new state.
    """

    _default_params: Any

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        assert population_size > 0, "Population size must be greater than 0"

        self.population_size = population_size
        self.solution = solution
        self.metrics_fn = metrics_fn

        self._ravel_solution, self._unravel_solution = get_ravel_fn(solution)
        self.solution_flat = self._ravel_solution(solution)
        self.num_dims = self.solution_flat.size

        # Fitness shaping function
        self.fitness_shaping_fn = fitness_shaping_fn

        # Default elite ratio
        self.elite_ratio = 1.0

        # Maximum num_dims that prevents overflow of num_dims**2 in int32
        self.max_num_dims_sq = jnp.minimum(
            self.num_dims, jnp.floor(jnp.sqrt(jnp.iinfo(jnp.int32).max))
        )

    @property
    def num_elites(self):
        """Number of elites, `max(1, int(elite_ratio * population_size))`."""
        return max(1, int(self.elite_ratio * self.population_size))

    @property
    def default_params(self) -> Any:
        """Default hyperparameters of the algorithm.

        Returns a `Params` dataclass. Use `params.replace(...)` to override
        individual hyperparameters before passing it to `init`, `ask` and `tell`.
        """
        return self._default_params

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        params: Any,
    ) -> Any:
        """Initialize the algorithm state.

        Args:
            key: PRNG key.
            params: Algorithm hyperparameters, e.g. `default_params`.

        Returns:
            The initial algorithm state.
        """
        state = self._init(key, params)
        return state

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self,
        key: jax.Array,
        state: Any,
        params: Any,
    ) -> tuple[Population, Any]:
        """Ask for a population of candidate solutions to evaluate.

        Args:
            key: PRNG key.
            state: Current algorithm state.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(population, state)`. `population` is a PyTree with the
            structure of `solution` and a leading axis of size `population_size`.
        """
        # Generate population
        population, state = self._ask(key, state, params)

        # Unravel population
        population = jax.vmap(self._unravel_solution)(population)

        return population, state

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: Any,
        params: Any,
    ) -> tuple[Any, Metrics]:
        """Update the algorithm state with the fitness of a population.

        Args:
            key: PRNG key.
            population: Population returned by `ask`.
            fitness: Fitness of each population member, shape
                `(population_size,)`. Lower is better. NaN values mark failed
                evaluations and never become the best solution.
            state: Algorithm state returned by `ask`.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(state, metrics)` with the updated state and the output of
            `metrics_fn`.
        """
        # Ravel population
        population = jax.vmap(self._ravel_solution)(population)

        # Update best solution and fitness
        best_solution, best_fitness = update_best_solution_and_fitness(
            population, fitness, state.best_solution, state.best_fitness
        )
        state = state.replace(best_solution=best_solution, best_fitness=best_fitness)

        # Compute metrics
        metrics = self.metrics_fn(key, population, fitness, state, params)

        # Shape fitness
        raw_fitness = fitness
        fitness = self.fitness_shaping_fn(population, raw_fitness, state, params)
        fitness = self._postprocess_fitness(raw_fitness, fitness)

        # Update state
        state = self._tell(key, population, fitness, state, params)
        state = state.replace(generation_counter=state.generation_counter + 1)

        return state, metrics

    def _init(self, key: jax.Array, params: Any) -> Any:
        raise NotImplementedError

    def _ask(
        self,
        key: jax.Array,
        state: Any,
        params: Any,
    ) -> tuple[Population, Any]:
        raise NotImplementedError

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: Any,
        params: Any,
    ) -> Any:
        raise NotImplementedError

    def _postprocess_fitness(self, raw_fitness: Fitness, fitness: Fitness) -> Fitness:
        """Apply algorithm-specific processing after fitness shaping."""
        return fitness


def get_ravel_fn(solution: Solution):
    """Return functions to flatten and reconstruct a PyTree solution.

    Args:
        solution: Example solution PyTree defining the structure to reconstruct.

    Returns:
        A tuple `(ravel_solution, unravel_solution)`: the first maps a PyTree to a
        flat vector, the second maps a flat vector back to a PyTree.
    """

    def ravel_solution(solution):
        flat, _ = flatten_util.ravel_pytree(solution)
        return flat

    _, unravel_solution = flatten_util.ravel_pytree(solution)

    return ravel_solution, unravel_solution


def update_best_solution_and_fitness(
    population, fitness, best_solution_so_far, best_fitness_so_far
):
    """Update best solution and fitness so far.

    Args:
        population: Flat population of shape `(population_size, num_dims)`.
        fitness: Fitness values of shape `(population_size,)`. NaN values are
            treated as failed evaluations and cannot improve the best solution.
        best_solution_so_far: Best solution found before this generation.
        best_fitness_so_far: Best fitness value found before this generation.

    Returns:
        A tuple `(best_solution, best_fitness)` seen so far.
    """
    # Failed evaluations cannot improve the archive.
    archive_fitness = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
    idx = jnp.argmin(archive_fitness)
    best_solution_in_population = population[idx]
    best_fitness_in_population = archive_fitness[idx]

    condition = best_fitness_in_population < best_fitness_so_far
    best_solution_so_far = jnp.where(
        condition[..., None],
        best_solution_in_population,
        best_solution_so_far,
    )
    best_fitness_so_far = jnp.where(
        condition,
        best_fitness_in_population,
        best_fitness_so_far,
    )
    return best_solution_so_far, best_fitness_so_far
