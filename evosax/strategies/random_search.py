"""Random Search."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


@struct.dataclass
class State(State):
    best_solution: Solution
    best_fitness: Fitness


@struct.dataclass
class Params(Params):
    pass


class RandomSearch(Strategy):
    """Random Search."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        sampling_fn: Callable,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Random Search."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "RandomSearch"
        self.sampling_fn = sampling_fn

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
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
        population = jax.vmap(self.sampling_fn)(keys)
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        return state
