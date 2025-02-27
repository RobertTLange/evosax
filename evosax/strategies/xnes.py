"""Exponential Natural Evolution Strategy (Wierstra et al., 2014).

Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
Inspired by: https://github.com/chanshing/xnes
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution
from .snes import get_weights as get_snes_weights


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    B: jax.Array
    lrate_std: float
    weights: jax.Array
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    lrate_mean: float
    lrate_std_init: float
    lrate_B: float
    use_adaptation_sampling: bool  # Adaptation sampling lrate std
    rho: float  # Significance level adaptation sampling
    c_prime: float  # Adaptation sampling step size


class xNES(Strategy):
    """Exponential Natural Evolution Strategy (xNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize xNES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "xNES"

    @property
    def _default_params(self) -> Params:
        lrate_std_init = (9 + 3 * jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims) * self.num_dims
        )
        rho = 0.5 - 1 / (3 * (self.num_dims + 1))
        return Params(
            std_init=1.0,
            lrate_mean=1.0,
            lrate_std_init=lrate_std_init,
            lrate_B=0.1,
            use_adaptation_sampling=False,
            rho=rho,
            c_prime=0.1,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        weights = get_snes_weights(self.population_size)

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            B=params.std_init
            * jnp.eye(self.num_dims),  # TODO: check if this is correct
            lrate_std=params.lrate_std_init,
            weights=weights,
            z=jnp.zeros((self.population_size, self.num_dims)),
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
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * jax.vmap(jnp.matmul, in_axes=(None, 0))(
            state.B.T, z
        )
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z_sorted = state.z[fitness.argsort()]
        weights = jnp.expand_dims(state.weights, axis=-1)

        # Update mean
        grad_mean = jnp.sum(weights * z_sorted, axis=0)
        mean = state.mean + params.lrate_mean * state.std * state.B @ grad_mean

        # Update std
        grad_M = jnp.sum(
            jnp.expand_dims(weights, axis=-1)
            * (jax.vmap(jnp.outer)(z_sorted, z_sorted) - jnp.eye(self.num_dims)),
            axis=0,
        )
        grad_std = jnp.trace(grad_M) / self.num_dims
        std = state.std * jnp.exp(0.5 * state.lrate_std * grad_std)

        # Update B
        grad_B = grad_M - grad_std * jnp.eye(self.num_dims)
        B = state.B * jnp.exp(0.5 * params.lrate_B * grad_B)

        # Adaptation sampling
        lrate_std = adaptation_sampling(
            state.lrate_std,
            params.lrate_std_init,
            mean,
            B,
            std,
            state.std,
            z_sorted,
            params.c_prime,
            params.rho,
        )
        lrate_std = jax.lax.select(
            params.use_adaptation_sampling, lrate_std, state.lrate_std
        )

        return state.replace(mean=mean, std=std, B=B, lrate_std=lrate_std)


def adaptation_sampling(
    lrate_std: float,
    lrate_std_init: float,
    mean: jax.Array,
    B: jax.Array,
    std: float,
    std_old: float,
    sorted_noise: jax.Array,
    c_prime: float,
    rho: float,
) -> float:
    """Adaptation sampling on std/std learning rate."""
    BB = B.T @ B
    A = std**2 * BB
    std_prime = std * jnp.sqrt(std / std_old)
    A_prime = std_prime**2 * BB

    # Probability ration and u-test - sorted order assumed for noise
    prob_0 = jax.scipy.stats.multivariate_normal.logpdf(sorted_noise, mean, A)
    prob_1 = jax.scipy.stats.multivariate_normal.logpdf(sorted_noise, mean, A_prime)
    w = jnp.exp(prob_1 - prob_0)
    population_size = sorted_noise.shape[0]
    n = jnp.sum(w)
    u = jnp.sum(w * (jnp.arange(population_size) + 0.5))
    u_mean = population_size * n / 2
    u_std = jnp.sqrt(population_size * n * (population_size + n + 1) / 12)
    cumulative = jax.scipy.stats.norm.cdf(u, loc=u_mean + 1e-10, scale=u_std + 1e-10)

    # Check test significance and update lrate
    lrate_std = jax.lax.select(
        cumulative < rho,
        (1 - c_prime) * lrate_std + c_prime * lrate_std_init,
        jnp.minimum(1, (1 - c_prime) * lrate_std),
    )
    return lrate_std
