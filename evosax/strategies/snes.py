import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy
from .des import get_des_weights


@struct.dataclass
class State:
    mean: chex.Array
    sigma: chex.Array
    weights: chex.Array
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    lrate_mean: float = 1.0
    lrate_sigma: float = 1.0
    sigma_init: float = 1.0
    temperature: float = 0.0
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


def get_snes_weights(population_size: int, use_baseline: bool = True):
    """Get recombination weights for different ranks."""

    def get_weight(i):
        return jnp.maximum(0, jnp.log(population_size / 2 + 1) - jnp.log(i))

    weights = jax.vmap(get_weight)(jnp.arange(1, population_size + 1))
    weights_norm = weights / jnp.sum(weights)
    return (weights_norm - use_baseline * (1 / population_size))[:, None]


class SNES(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: chex.ArrayTree | chex.Array | None = None,
        sigma_init: float = 1.0,
        temperature: float = 0.0,  # good values tend to be between 12 and 20
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Separable Exponential Natural ES (Wierstra et al., 2014)
        Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
        """
        super().__init__(population_size, solution, mean_decay, **fitness_kwargs)
        self.strategy_name = "SNES"

        # Set core kwargs params
        self.sigma_init = sigma_init
        self.temperature = temperature

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolutionary strategy."""
        lrate_sigma = (3 + jnp.log(self.num_dims)) / (5 * jnp.sqrt(self.num_dims))
        params = Params(
            lrate_sigma=lrate_sigma,
            sigma_init=self.sigma_init,
            temperature=self.temperature,
        )
        return params

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolutionary strategy."""
        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        use_des_weights = params.temperature > 0.0
        weights = jax.lax.select(
            use_des_weights,
            get_des_weights(self.population_size, params.temperature),
            get_snes_weights(self.population_size),
        )
        state = State(
            mean=initialization,
            sigma=params.sigma_init * jnp.ones(self.num_dims),
            weights=weights,
            best_member=initialization,
        )

        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[chex.Array, State]:
        """`ask` for new parameter candidates to evaluate next."""
        noise = jax.random.normal(key, (self.population_size, self.num_dims))
        x = state.mean + noise * state.sigma.reshape(1, self.num_dims)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: State,
        params: Params,
    ) -> State:
        """`tell` performance data for strategy state update."""
        s = (x - state.mean) / state.sigma
        ranks = fitness.argsort()
        sorted_noise = s[ranks]
        grad_mean = (state.weights * sorted_noise).sum(axis=0)
        grad_sigma = (state.weights * (sorted_noise**2 - 1)).sum(axis=0)
        mean = state.mean + params.lrate_mean * state.sigma * grad_mean
        sigma = state.sigma * jnp.exp(params.lrate_sigma / 2 * grad_sigma)
        return state.replace(mean=mean, sigma=sigma)
