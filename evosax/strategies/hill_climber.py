import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from ..types import Fitness, Population, Solution
from ..utils import get_best_fitness_member


@struct.dataclass
class State:
    mean: jax.Array
    sigma: jax.Array
    best_member: jax.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    sigma_init: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class HillClimber(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Simple Gaussian Hill Climbing"""
        super().__init__(population_size, solution, mean_decay, **fitness_kwargs)
        self.strategy_name = "HillClimber"

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        return Params()

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolution strategy."""
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones((self.num_dims,)),
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[jax.Array, State]:
        """`ask` for new proposed candidates to evaluate next."""
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            key,
            (self.population_size, self.num_dims),
        )
        x = state.best_member + state.sigma.reshape(1, self.num_dims) * z
        return x, state

    def tell_strategy(
        self,
        x: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        """`tell` update to ES state."""
        best_member, best_fitness = get_best_fitness_member(x, fitness, state, False)
        return state.replace(mean=best_member)
