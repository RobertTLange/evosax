import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from ..types import Fitness, Population, Solution


@struct.dataclass
class State:
    mean: jax.Array
    best_member: jax.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    init_min: float = 0.0
    init_max: float = 0.0
    range_min: float = 0.0
    range_max: float = 1.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class RandomSearch(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        **fitness_kwargs: bool | int | float,
    ):
        """Simple Random Search Baseline"""
        super().__init__(population_size, solution, **fitness_kwargs)
        self.strategy_name = "RandomSearch"

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        return Params()

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the differential evolution strategy."""
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            mean=initialization,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[jax.Array, State]:
        """`ask` for new proposed candidates to evaluate next."""
        x = jax.random.uniform(
            key,
            (self.population_size, self.num_dims),
            minval=params.range_min,
            maxval=params.range_max,
        )
        return x, state

    def tell_strategy(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """`tell` update to ES state."""
        idx = jnp.argsort(fitness)
        fitness = fitness[idx]
        x = x[idx]
        # Set mean to best member seen so far
        improved = fitness[0] < state.best_fitness
        best_mean = jax.lax.select(improved, x[0], state.best_member)
        return state.replace(mean=best_mean)
