from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from ..strategy import Strategy
from ..utils import get_best_fitness_member


@struct.dataclass
class EvoState:
    mean: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    radius_max: float = 0.2
    radius_min: float = 0.001
    radius_decay: float = 5
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class GLD(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Gradientless Descent (Golovin et al., 2019)
        Reference: https://arxiv.org/pdf/1911.06317.pdf"""
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        self.strategy_name = "GLD"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams()

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            rng,
            (self.popsize, self.num_dims),
        )
        # Exponentially decaying sigma scale
        sigma_scale = params.radius_min + jnp.exp2(
            -jnp.arange(self.popsize) / params.radius_decay
        ) * (params.radius_max - params.radius_min)
        sigma_scale = sigma_scale.reshape(-1, 1)
        # print(state["best_member"].shape, (sigma_scale * z).shape)
        x = state.best_member + sigma_scale * z
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
