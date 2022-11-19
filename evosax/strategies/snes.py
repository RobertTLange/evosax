import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from flax import struct


@struct.dataclass
class EvoState:
    mean: chex.Array
    sigma: chex.Array
    weights: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    lrate_mean: float = 1.0
    lrate_sigma: float = 1.0
    sigma_init: float = 1.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def get_recombination_weights(popsize: int, use_baseline: bool = True):
    """Get recombination weights for different ranks."""

    def get_weight(i):
        return jnp.maximum(0, jnp.log(popsize / 2 + 1) - jnp.log(i))

    weights = jax.vmap(get_weight)(jnp.arange(1, popsize + 1))
    weights_norm = weights / jnp.sum(weights)
    return weights_norm - use_baseline * (1 / popsize)


class SNES(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Exponential Natural ES (Wierstra et al., 2014)
        Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        """
        super().__init__(num_dims, popsize)
        self.strategy_name = "SNES"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolutionary strategy."""
        lrate_sigma = (3 + jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims)
        )
        params = EvoParams(lrate_sigma=lrate_sigma)
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolutionary strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        weights = get_recombination_weights(self.popsize)
        state = EvoState(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones(self.num_dims),
            weights=weights.reshape(-1, 1),
            best_member=initialization,
        )

        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        noise = jax.random.normal(rng, (self.popsize, self.num_dims))
        x = state.mean + noise * state.sigma.reshape(1, self.num_dims)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        s = (x - state.mean) / state.sigma
        ranks = fitness.argsort()
        sorted_noise = s[ranks]
        grad_mean = (state.weights * sorted_noise).sum(axis=0)
        grad_sigma = (state.weights * (sorted_noise ** 2 - 1)).sum(axis=0)
        mean = state.mean + params.lrate_mean * state.sigma * grad_mean
        sigma = state.sigma * jnp.exp(params.lrate_sigma / 2 * grad_sigma)
        return state.replace(mean=mean, sigma=sigma)
