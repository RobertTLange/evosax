"""1/5 Mutation Rate Genetic Algorithm (Rechenberg, 1987).

Reference: https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import Params, PopulationBasedAlgorithm, State, metrics_fn
from .simple_ga import mutation


@struct.dataclass
class State(State):
    population: jax.Array
    fitness: Fitness
    std: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_limit: float
    std_ratio: float


class MR15_GA(PopulationBasedAlgorithm):
    """1/5 Mutation Rate Genetic Algorithm (MR15-GA)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize MR15-GA."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            std_limit=0.001,
            std_ratio=0.2,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            std=params.std_init,
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
        # Mutation
        keys = jax.random.split(key, self.population_size)
        population = jax.vmap(mutation, in_axes=(0, 0, None))(
            keys, state.population, state.std
        )
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update mutation std
        beneficial_mutation_rate = jnp.mean(fitness < state.fitness)
        increase_std = beneficial_mutation_rate > params.std_ratio
        std = jnp.where(increase_std, 2 * state.std, 0.5 * state.std)
        std = jnp.clip(std, min=params.std_limit)

        # Combine populations from current and previous generations
        population = jnp.concatenate([population, state.population])
        fitness = jnp.concatenate([fitness, state.fitness])

        # Select top elite from total population info
        idx = jnp.argsort(fitness)[: self.population_size]
        population = population[idx]
        fitness = fitness[idx]

        return state.replace(population=population, fitness=fitness, std=std)
