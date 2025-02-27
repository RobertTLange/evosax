"""Matrix Adaptation Evolution Strategy (Bayer & Sendhoff, 2017).

Reference: https://ieeexplore.ieee.org/document/7875115
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    p_std: jax.Array
    M: jax.Array
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    c_1: float
    c_mu: float
    chi_n: float


class MA_ES(Strategy):
    """Matrix Adaptation Evolution Strategy (MA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize MA-ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "MA_ES"

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )  # Eq. (48)

        mu_eff = (jnp.sum(weights_prime[: self.num_elites]) ** 2) / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )  # Eq. (8)
        mu_eff_minus = (jnp.sum(weights_prime[self.num_elites :]) ** 2) / jnp.sum(
            weights_prime[self.num_elites :] ** 2
        )  # Table 1

        # Decay rate for cumulation path
        c_c = (4 + mu_eff / self.num_dims) / (
            self.num_dims + 4 + 2 * mu_eff / self.num_dims
        )  # Eq. (56)

        # Learning rate rate for rank-one update
        alpha_cov = 2
        c_1 = alpha_cov / ((self.max_num_dims_sq + 1.3) ** 2 + mu_eff)  # Eq. (57)

        # Learning rate for rank-mu update
        c_mu = jnp.minimum(
            1 - c_1 - 1e-8,  # 1e-8 is for large population size
            alpha_cov
            * (mu_eff + 1 / mu_eff - 2)
            / ((self.max_num_dims_sq + 2) ** 2 + alpha_cov * mu_eff / 2),
        )  # Eq. (58)

        # Minimum alpha
        min_alpha = jnp.minimum(
            1 + c_1 / c_mu,  # Eq. (50)
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # Eq. (51)
        )
        min_alpha = jnp.minimum(
            min_alpha,
            (1 - c_1 - c_mu) / (self.num_dims * c_mu),  # Eq. (52)
        )

        # Eq. (53)
        positive_sum = jnp.sum(weights_prime * (weights_prime > 0))
        negative_sum = jnp.sum(jnp.abs(weights_prime * (weights_prime < 0)))
        weights = jnp.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            0.0,
        )

        # Learning rate for mean
        c_mean = 1.0  # Eq. (54)

        # Step-size control
        c_std = (mu_eff + 2) / (self.num_dims + mu_eff + 5)  # Eq. (55)
        d_std = (
            1
            + 2 * jnp.maximum(0, jnp.sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
            + c_std
        )  # Eq. (55)
        chi_n = jnp.sqrt(self.num_dims) * (
            1.0
            - (1.0 / (4.0 * self.num_dims))
            + 1.0 / (21.0 * (self.max_num_dims_sq**2))
        )  # Page 28

        return Params(
            std_init=1.0,
            weights=weights,
            mu_eff=mu_eff,
            c_mean=c_mean,
            c_std=c_std,
            d_std=d_std,
            c_1=c_1,
            c_mu=c_mu,
            chi_n=chi_n,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            M=jnp.eye(self.num_dims),
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
        d = jax.vmap(jnp.matmul, in_axes=(None, 0))(state.M, z)
        population = state.mean + state.std * d
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Sort
        idx = jnp.argsort(fitness)
        state = state.replace(z=state.z[idx])

        # Update mean
        mean = state.mean + params.c_mean * state.std * state.M @ jnp.dot(
            params.weights, state.z
        )

        # Update evolution path
        p_std = (1 - params.c_std) * state.p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * jnp.dot(params.weights, state.z)

        # Update M matrix
        p_std_outer = jnp.outer(p_std, p_std)
        z_outer_w = jnp.einsum(
            "i,ijk->jk", params.weights, jax.vmap(jnp.outer)(state.z, state.z)
        )
        I = jnp.eye(self.num_dims)
        M = state.M @ (
            I + params.c_1 / 2 * (p_std_outer - I) + params.c_mu / 2 * (z_outer_w - I)
        )

        # Update std
        std = state.std * jnp.exp(
            (params.c_std / params.d_std) * (jnp.linalg.norm(p_std) / params.chi_n - 1)
        )
        return state.replace(mean=mean, std=std, p_std=p_std, M=M)
