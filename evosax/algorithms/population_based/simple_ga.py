"""Simple Genetic Algorithm (Such et al., 2017).

[1] https://arxiv.org/abs/1712.06567
[2] https://github.com/hardmaru/estool/blob/master/es.py
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import Params, PopulationBasedAlgorithm, State, metrics_fn


@struct.dataclass
class State(State):
    population: Population
    fitness: Fitness
    std: jax.Array


@struct.dataclass
class Params(Params):
    crossover_rate: float


class SimpleGA(PopulationBasedAlgorithm):
    """Simple Genetic Algorithm (Simple GA)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize Simple GA."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params(crossover_rate=0.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            std=self.std_schedule(0),
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
        # Get elites
        idx = jnp.argsort(state.fitness)
        population = state.population[idx]
        p = jnp.arange(self.population_size) < self.num_elites

        key_crossover, key_mutation, key_1, key_2 = jax.random.split(key, 4)
        key_crossover = jax.random.split(key_crossover, self.population_size)
        key_mutation = jax.random.split(key_mutation, self.population_size)

        # Crossover
        parents_1 = jax.random.choice(key_1, population, (self.population_size,), p=p)
        parents_2 = jax.random.choice(key_2, population, (self.population_size,), p=p)

        population = jax.vmap(crossover, in_axes=(0, 0, 0, None))(
            key_crossover, parents_1, parents_2, params.crossover_rate
        )

        # Mutation
        population = jax.vmap(mutation, in_axes=(0, 0, None))(
            key_mutation, population, state.std
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
        return state.replace(
            population=population,
            fitness=fitness,
            std=self.std_schedule(state.generation_counter),
        )


def crossover(
    key: jax.Array, parent_1: Solution, parent_2: Solution, crossover_rate: float
) -> Solution:
    """Crossover between two parents."""
    mask = jax.random.uniform(key, parent_1.shape) < crossover_rate
    return parent_1 * (1 - mask) + parent_2 * mask


def mutation(key: jax.Array, solution: Solution, std: jax.Array) -> Solution:
    """Mutation of a solution."""
    return solution + std * jax.random.normal(key, solution.shape)
