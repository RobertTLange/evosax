import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Strategy


@struct.dataclass
class State:
    mean: chex.Array
    sigma: chex.Array
    weights: chex.Array  # Weights for population members
    best_member: chex.Array
    best_fitness: float = jnp.finfo(jnp.float32).max
    generation_counter: int = 0


@struct.dataclass
class Params:
    c_sigma: float = 0.1  # Learning rate for population std
    c_m: float = 1.0  # Learning rate for population mean
    sigma_init: float = 1.0  # Standard deviation
    init_min: float = 0.0
    init_max: float = 0.0
    clip_min: float = -jnp.finfo(jnp.float32).max
    clip_max: float = jnp.finfo(jnp.float32).max


class SimpleES(Strategy):
    def __init__(
        self,
        population_size: int,
        solution: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Simple Gaussian Evolution Strategy (Rechenberg, 1975)
        Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
        """
        super().__init__(population_size, solution, mean_decay, **fitness_kwargs)
        self.elite_ratio = elite_ratio
        self.elite_population_size = max(
            1, int(self.population_size * self.elite_ratio)
        )
        self.strategy_name = "SimpleES"

        # Set core kwargs params
        self.sigma_init = sigma_init

    @property
    def params_strategy(self) -> Params:
        """Return default parameters of evolution strategy."""
        # Only parents have positive weight - equal weighting!
        return Params(sigma_init=self.sigma_init)

    def init_strategy(self, key: jax.Array, params: Params) -> State:
        """`init` the evolution strategy."""
        weights = jnp.zeros(self.population_size)
        weights = weights.at[: self.elite_population_size].set(
            1 / self.elite_population_size
        )

        initialization = jax.random.uniform(
            key,
            (self.num_dims,),
            minval=params.init_min,
            maxval=params.init_max,
        )
        state = State(
            mean=initialization,
            sigma=jnp.repeat(params.sigma_init, self.num_dims),
            weights=weights,
            best_member=initialization,
        )
        return state

    def ask_strategy(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[chex.Array, State]:
        """`ask` for new proposed candidates to evaluate next."""
        z = jax.random.normal(key, (self.population_size, self.num_dims))  # ~ N(0, I)
        x = state.mean + state.sigma * z  # ~ N(m, σ^2 I)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: State,
        params: Params,
    ) -> State:
        """`tell` update to ES state."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, y_k = update_mean(sorted_solutions, state.mean, params.c_m, state.weights)
        sigma = update_sigma(y_k, state.sigma, params.c_sigma, state.weights)
        return state.replace(mean=mean, sigma=sigma)


def update_mean(
    sorted_solutions: chex.Array,
    mean: chex.Array,
    c_m: float,
    weights: chex.Array,
) -> tuple[chex.Array, chex.Array]:
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
    sigma_est = jnp.sqrt(jnp.sum((y_k.T**2 * weights), axis=1))
    sigma_new = (1 - c_sigma) * sigma + c_sigma * sigma_est
    return sigma_new
