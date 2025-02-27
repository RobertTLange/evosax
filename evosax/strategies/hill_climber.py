"""Gaussian Hill Climbing algorithm."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


@struct.dataclass
class State(State):
    mean: jax.Array
    fitness: float
    std: jax.Array
    generation_counter: int


@struct.dataclass
class Params(Params):
    std_init: float
    std_decay: float
    std_limit: float


class HillClimber(Strategy):
    """Gaussian Hill Climbing algorithm."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Gaussian Hill Climbing."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "HillClimber"

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            std_decay=1.0,
            std_limit=0.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            fitness=jnp.inf,
            std=params.std_init * jnp.ones((self.num_dims,)),
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
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + jnp.expand_dims(state.std, axis=0) * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Get best member in current population
        best_idx = jnp.argmin(fitness)
        best_member, best_fitness = population[best_idx], fitness[best_idx]

        # Replace if new best
        replace = best_fitness < state.fitness
        best_member = jnp.where(replace, best_member, state.mean)
        best_fitness = jnp.where(replace, best_fitness, state.fitness)

        # Update std
        std = jnp.clip(state.std * params.std_decay, min=params.std_limit)

        return state.replace(mean=best_member, fitness=best_fitness, std=std)
