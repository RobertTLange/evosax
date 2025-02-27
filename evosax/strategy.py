"""Base module for evolution strategy."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from .core import FitnessShaper
from .types import Fitness, Metrics, Population, Solution


@struct.dataclass
class State:
    best_solution: Solution
    best_fitness: float
    generation_counter: int


@struct.dataclass
class Params:
    pass


def metrics_fn(key, population, fitness, state, params) -> Metrics:
    """Compute metrics for evolution strategy."""
    idx = jnp.argmin(fitness)
    return {"best_fitness": fitness[idx], "best_solution": population[idx]}


class Strategy:
    """Base class for evolution strategy."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize base class for evolution strategy."""
        assert population_size > 0, "Population size must be greater than 0"

        self.population_size = population_size
        self.solution = solution
        self.metrics_fn = metrics_fn

        self._ravel_solution, self._unravel_solution = get_ravel_fn(solution)
        self.solution_flat = self._ravel_solution(solution)
        self.num_dims = self.solution_flat.size

        # Setup optional fitness shaper
        self.fitness_shaper = FitnessShaper(**fitness_kwargs)

        # Default elite ratio
        self.elite_ratio = 1.0

        # Maximum num_dims that prevents overflow of num_dims**2 in int32
        self.max_num_dims_sq = jnp.minimum(
            self.num_dims, jnp.floor(jnp.sqrt(jnp.iinfo(jnp.int32).max))
        )

    @property
    def num_elites(self):
        """Set the elite ratio and update num_elites."""
        return max(1, int(self.elite_ratio * self.population_size))

    @property
    def default_params(self) -> Params:
        """Return default evolution strategy params."""
        return self._default_params

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        params: Params,
    ) -> State:
        """Initialize evolution strategy."""
        state = self._init(key, params)
        return state

    @partial(jax.jit, static_argnames=("self",))
    def ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        """Ask evolution strategy for new candidate solutions to evaluate."""
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
        state: State,
        params: Params,
    ) -> tuple[State, Metrics]:
        """Tell evolution strategy fitness for state update."""
        # Update best solution and fitness
        best_solution, best_fitness = update_best_solution_and_fitness(
            population, fitness, state.best_solution, state.best_fitness
        )
        state = state.replace(best_solution=best_solution, best_fitness=best_fitness)

        # Compute metrics
        metrics = self.metrics_fn(key, population, fitness, state, params)

        # Ravel population
        population = jax.vmap(self._ravel_solution)(population)

        # Shape fitness
        fitness = self.fitness_shaper.apply(population, fitness)

        # Update state
        state = self._tell(key, population, fitness, state, params)
        state = state.replace(generation_counter=state.generation_counter + 1)

        return state, metrics

    def _init(
        self,
        key: jax.Array,
        solution: Solution,
        fitness: Fitness,
        params: Params,
    ) -> State:
        raise NotImplementedError

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        raise NotImplementedError

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        raise NotImplementedError

    def get_eval_solution(self, state: State) -> Solution:
        """Return reshaped parameters to evaluate."""
        solution = self._unravel_solution(state.mean)
        return solution

    def set_mean(self, state: State, solution: Solution) -> State:
        """Update state with new mean."""
        mean = self._ravel_solution(solution)
        state = state.replace(mean=mean)
        return state


def get_ravel_fn(solution: Solution):
    """Return functions to flatten and reconstruct a PyTree solution."""

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
        population: Array of solutions
        fitness: Array of fitness values
        best_solution_so_far: Best solution found before this generation
        best_fitness_so_far: Best fitness value found before this generation

    Returns:
        tuple: containing the best solution and fitness seen so far.

    """
    idx = jnp.argmin(fitness)
    best_solution_in_population = population[idx]
    best_fitness_in_population = fitness[idx]

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
