import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from ..utils import get_best_fitness_member


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    sigma_init: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class HillClimber(Strategy):
    def __init__(
        self,
        popsize: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Simple Gaussian Hill Climbing"""
        super().__init__(popsize, pholder_params, mean_decay, **fitness_kwargs)
        self.strategy_name = "HillClimber"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(self, key: jax.Array, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones((self.num_dims,)),
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: EvoState, params: EvoParams
    ) -> tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            key,
            (self.popsize, self.num_dims),
        )
        x = state.best_member + state.sigma.reshape(1, self.num_dims) * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        best_member, best_fitness = get_best_fitness_member(x, fitness, state, False)
        return state.replace(mean=best_member)
