"""Differential Evolution (Storn & Price, 1997).

Reference: https://link.springer.com/article/10.1023/A:1008202821328
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...types import Fitness, Population, Solution
from .base import Params, PopulationBasedAlgorithm, State, metrics_fn


@struct.dataclass
class State(State):
    population: jax.Array
    fitness: jax.Array


@struct.dataclass
class Params(Params):
    elitism: bool  # If elitism, base vector is best member else random
    crossover_rate: float  # [0, 1]
    differential_weight: float  # [0, 2]


class DE(PopulationBasedAlgorithm):
    """Differential Evolution (DE)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        num_diff: int = 1,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize DE."""
        assert population_size >= 4, "DE requires population_size >= 4."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.num_diff = num_diff

    @property
    def _default_params(self) -> Params:
        return Params(
            elitism=True,
            crossover_rate=0.9,
            differential_weight=0.8,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            population=jnp.full((self.population_size, self.num_dims), jnp.nan),
            fitness=jnp.full(self.population_size, jnp.inf),
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
        keys = jax.random.split(key, self.population_size)
        member_ids = jnp.arange(self.population_size)
        best_index = jnp.argmin(state.fitness)

        def _ask_member(key, member_id):
            x = state.population[member_id]

            key_a, key_R, key_r, key_ab = jax.random.split(key, 4)
            p = jnp.ones(self.population_size).at[member_id].set(0.0)

            # Base vector
            a_index = jax.random.choice(key_a, self.population_size, p=p)

            # Elitism
            a_index = jnp.where(params.elitism, best_index, a_index)
            a = state.population[a_index]

            # Crossover dimensions mask
            R = jax.random.choice(key_R, self.num_dims)
            R = jax.nn.one_hot(R, self.num_dims)

            r = jax.random.uniform(key_r, (self.num_dims,))

            mask = jnp.logical_or(r < params.crossover_rate, R)

            # Diff vectors
            p = p.at[a_index].set(0.0)
            for _ in range(self.num_diff):
                key_ab, subkey = jax.random.split(key_ab)
                b, c = jax.random.choice(
                    subkey, state.population, (2,), replace=False, p=p
                )

                a = jnp.where(mask, a + params.differential_weight * (b - c), x)

            return a

        y = jax.vmap(_ask_member)(keys, member_ids)
        return y, state

    def _tell(
        self,
        key: jax.Array,
        population: Solution,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Replace member in population if performance improved
        replace = fitness <= state.fitness
        population = jnp.where(replace[..., None], population, state.population)
        fitness = jnp.where(replace, fitness, state.fitness)
        return state.replace(population=population, fitness=fitness)
