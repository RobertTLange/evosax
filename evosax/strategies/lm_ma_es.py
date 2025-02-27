"""Limited Memory Matrix Adaptation Evolution Strategy (Loshchilov et al., 2017).

Reference: https://arxiv.org/abs/1705.06693
Note: The original paper recommends a population size of 4 + 3 * jnp.log(num_dims).
Instabilities have been observed with larger population sizes.
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
    d: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    chi_n: float
    c_c: jax.Array
    c_d: jax.Array


class LM_MA_ES(Strategy):
    """Limited Memory Matrix Adaptation Evolution Strategy (LM-MA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize LM-MA-ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "LM_MA_ES"

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        self.m = int(4 + jnp.floor(3 * jnp.log(self.num_dims)))

        std_init = 0.05

        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )  # Eq. (48)

        mu_eff = (jnp.sum(weights_prime[: self.num_elites]) ** 2) / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )  # Eq. (8)

        # Eq. (53)
        positive_sum = jnp.sum(weights_prime * (weights_prime > 0))
        weights = jnp.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            0.0,
        )

        # Learning rate for mean
        c_mean = 1.0  # Eq. (54)

        # Step-size control
        c_std = 2 * self.population_size / self.num_dims
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

        c_d = 1 / jnp.power(1.5, jnp.arange(self.m)) / self.num_dims
        c_c = self.population_size / jnp.power(4, jnp.arange(self.m)) / self.num_dims

        return Params(
            std_init=std_init,
            weights=weights,
            mu_eff=mu_eff,
            c_mean=c_mean,
            c_std=c_std,
            d_std=d_std,
            chi_n=chi_n,
            c_d=c_d,
            c_c=c_c,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            M=jnp.zeros((self.m, self.num_dims)),
            z=jnp.zeros((self.population_size, self.num_dims)),
            d=jnp.zeros((self.population_size, self.num_dims)),
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

        d = z
        # L9-10
        for i in range(self.m):
            d = jnp.where(
                i < state.generation_counter,
                (1 - params.c_d[i]) * d
                + params.c_d[i] * jnp.dot(d, state.M[i])[:, None] * state.M[i],
                d,
            )

        population = state.mean + state.std * d
        return population, state.replace(z=z, d=d)

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
        state = state.replace(z=state.z[idx], d=state.d[idx])

        # Update mean
        mean = state.mean + params.c_mean * state.std * jnp.dot(params.weights, state.d)
        # mean = state.mean + state.c_mean * jnp.dot(params.weights, population[idx] - state.mean)

        # Update evolution path
        p_std = (1 - params.c_std) * state.p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * jnp.dot(params.weights, state.z)

        # Update m
        M = (1 - params.c_c)[:, None] * state.M + jnp.sqrt(
            params.mu_eff * params.c_c * (2 - params.c_c)
        )[:, None] * jnp.dot(params.weights, state.z)

        # Update std
        std = state.std * jnp.exp(
            params.c_std * (jnp.linalg.norm(p_std) / params.chi_n - 1) / params.d_std
        )
        return state.replace(mean=mean, std=std, p_std=p_std, M=M)
