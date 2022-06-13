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
    weights: chex.Array  # Weights for population members
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    gen_counter: int = 0


@struct.dataclass
class EvoParams:
    c_sigma: float = 0.1  # Learning rate for population std
    c_m: float = 1.0  # Learning rate for population mean
    sigma_init: float = 1.0  # Standard deviation
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SimpleES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """Simple Gaussian Evolution Strategy (Rechenberg, 1975)
        Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.strategy_name = "SimpleES"

    @property
    def params_strategy(self) -> EvoParams:
        """Return default parameters of evolution strategy."""
        # Only parents have positive weight - equal weighting!
        return EvoParams()

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: EvoParams
    ) -> EvoState:
        """`initialize` the evolution strategy."""
        weights = jnp.zeros(self.popsize)
        weights = weights.at[: self.elite_popsize].set(1 / self.elite_popsize)

        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = EvoState(
            mean=initialization,
            sigma=jnp.repeat(params.sigma_init, self.num_dims),
            weights=weights,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: EvoState, params: EvoParams
    ) -> Tuple[chex.Array, EvoState]:
        """`ask` for new proposed candidates to evaluate next."""
        z = jax.random.normal(rng, (self.popsize, self.num_dims))  # ~ N(0, I)
        x = state.mean + state.sigma * z  # ~ N(m, σ^2 I)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: EvoState,
        params: EvoParams,
    ) -> EvoState:
        """`tell` update to ES state."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, y_k = update_mean(
            sorted_solutions, state.mean, params.c_m, state.weights
        )
        sigma = update_sigma(y_k, state.sigma, params.c_sigma, state.weights)
        return state.replace(mean=mean, sigma=sigma)


def update_mean(
    sorted_solutions: chex.Array,
    mean: chex.Array,
    c_m: float,
    weights: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = x_k - mean
    y_w = jnp.sum(y_k.T * weights, axis=1)
    mean_new = mean + c_m * y_w
    return mean_new, y_k


def update_sigma(
    y_k: chex.Array, sigma: chex.Array, c_sigma: float, weights: chex.Array
) -> chex.Array:
    """Update stepsize sigma."""
    sigma_est = jnp.sqrt(jnp.sum((y_k.T ** 2 * weights), axis=1))
    sigma_new = (1 - c_sigma) * sigma + c_sigma * sigma_est
    return sigma_new
