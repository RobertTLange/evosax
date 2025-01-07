from typing import Tuple, Optional, Union
import jax
import jax.numpy as jnp
import chex
from flax import struct
from flax import linen as nn
from ..strategy import Strategy


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    weights: chex.Array  # Weights for population members
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    temperature: float = 12.5  # Temperature for softmax weights
    lrate_sigma: float = 0.1  # Learning rate for population std
    lrate_mean: float = 1.0  # Learning rate for population mean
    sigma_init: float = 0.1  # Standard deviation
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def get_des_weights(popsize: int, temperature: float = 12.5):
    """Compute discovered recombination weights."""
    ranks = jnp.arange(popsize)
    ranks /= ranks.size - 1
    ranks = ranks - 0.5
    sigout = nn.sigmoid(temperature * ranks)
    weights = nn.softmax(-20 * sigout)
    return weights[:, None]


class DES(Strategy):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        temperature: float = 12.5,
        sigma_init: float = 0.1,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """Discovered Evolution Strategy (Lange et al., 2023)"""
        super().__init__(
            popsize, num_dims, pholder_params, mean_decay, n_devices, **fitness_kwargs
        )
        self.strategy_name = "DES"
        self.temperature = temperature
        self.sigma_init = sigma_init

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        return EvoParams(temperature=self.temperature, sigma_init=self.sigma_init)

    def initialize_strategy(self, rng: chex.PRNGKey, params: EvoParams) -> EvoState:
        """`initialize` the evolution strategy."""
        # Get DES discovered recombination weights.
        weights = get_des_weights(self.popsize, params.temperature)
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones(self.num_dims),
            weights=weights,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        z = jax.random.normal(rng, (self.popsize, self.num_dims))  # ~ N(0, I)
        x = state.mean + z * state.sigma.reshape(1, self.num_dims)  # ~ N(m, σ^2 I)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        weights = state.weights
        x = x[fitness.argsort()]
        # Weighted updates
        weighted_mean = (weights * x).sum(axis=0)
        weighted_sigma = jnp.sqrt((weights * (x - state.mean) ** 2).sum(axis=0) + 1e-06)
        mean = state.mean + params.lrate_mean * (weighted_mean - state.mean)
        sigma = state.sigma + params.lrate_sigma * (weighted_sigma - state.sigma)
        return state.replace(mean=mean, sigma=sigma)
