"""Base module for evolutionary algorithms."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import flatten_util

from ..core.fitness_shaping import identity_fitness_shaping_fn
from ..types import Fitness, Metrics, Params, Population, Solution, State


@struct.dataclass
class State(State):
    best_solution: Solution
    best_fitness: float
    generation_counter: int


@struct.dataclass
class Params(Params):
    pass


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Compute metrics for distribution-based algorithm."""
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
    """Base class for evolutionary algorithms."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize base class for evolutionary algorithm."""
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
        """Set the elite ratio and update num_elites."""
        return max(1, int(self.elite_ratio * self.population_size))

    @property
    def default_params(self) -> Params:
        """Return default evolutionary algorithm params."""
        return self._default_params

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        params: Params,
    ) -> State:
        """Initialize evolutionary algorithm."""
        state = self._init(key, params)
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
        population = jax.vmap(self._ravel_solution)(population)

        # Update best solution and fitness
        best_solution, best_fitness = update_best_solution_and_fitness(
            population, fitness, state.best_solution, state.best_fitness
        )
        state = state.replace(best_solution=best_solution, best_fitness=best_fitness)

        # Compute metrics
        metrics = self.metrics_fn(key, population, fitness, state, params)

        # Shape fitness
        fitness = self.fitness_shaping_fn(population, fitness, state, params)

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
