"""Self-Adaptation Mutation Rate Genetic Algorithm."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
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
    std_init: float
    std_min: float
    std_max: float
    std_meta: float
    std_best_limit: float


class SAMR_GA(PopulationBasedAlgorithm):
    """Self-Adaptation Mutation Rate Genetic Algorithm (SAMR-GA)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SAMR-GA."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            std_min=0.0,
            std_max=10.0,
            std_meta=2.0,
            std_best_limit=0.0001,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full((self.population_size,), jnp.inf),
            std=jnp.full((self.population_size,), params.std_init),
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
        key_idx, key_eps_std, key_eps_x = jax.random.split(key, 3)

        # Sort population by fitness
        idx = jnp.argsort(state.fitness)
        population, std = state.population[idx], state.std[idx]

        # Select elites for mutation
        idx = jax.random.choice(key_idx, self.num_elites, (self.population_size - 1,))

        # Compute std
        eps_std = jax.random.uniform(
            key_eps_std, (self.population_size,), minval=-1, maxval=1
        )
        std = jnp.concatenate([std[:1], std[idx]]) * params.std_meta**eps_std
        std = jnp.clip(std, min=params.std_min, max=params.std_max)

        # Mutation
        eps_x = jax.random.normal(key_eps_x, (self.population_size, self.num_dims))
        population = jnp.concatenate([population[:1], population[idx]])
        population += std[:, None] * eps_x

        return population, state.replace(std=std)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        return state.replace(population=population, fitness=fitness)
