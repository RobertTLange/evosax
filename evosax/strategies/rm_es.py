"""Rank-m Evolution Strategy (Li & Zhang, 2017).

Reference: https://ieeexplore.ieee.org/document/8080257
Note: The original paper recommends a population size of 4 + 3 * jnp.log(num_dims).
Instabilities have been observed with larger population sizes.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..core.fitness import ranksort
from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    p_c: jax.Array  # p
    cumulative_rank_rate: float  # s
    P: jax.Array
    t: jax.Array
    fitness_elites_sorted: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_limit: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    c_c: float
    c_cov: float
    T: int
    q_star: float


class Rm_ES(Strategy):
    """Rank-m Evolution Strategy (Rm-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        m: int = 1,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Rm-ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.m = m  # number of evolution paths
        self.strategy_name = "Rm_ES"

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )  # From CMA-ES paper, slightly modified from paper, works better

        mu_eff = (jnp.sum(weights_prime[: self.num_elites]) ** 2) / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )

        positive_sum = jnp.sum(weights_prime * (weights_prime > 0))
        weights = jnp.where(
            weights_prime >= 0,
            weights_prime / positive_sum,
            0.0,
        )

        params = Params(
            std_init=1.0,
            std_limit=1e-3,
            weights=weights,
            mu_eff=mu_eff,
            c_mean=1.0,
            c_std=0.3,
            d_std=1.0,
            c_c=2 / (self.num_dims + 7),  # Table 1
            c_cov=1 / (3 * jnp.sqrt(self.num_dims) + 5),  # Table 1
            T=20,
            q_star=0.3,
        )

        return params

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_c=jnp.zeros((self.num_dims,)),
            cumulative_rank_rate=0.0,
            P=jnp.zeros((self.m, self.num_dims)),
            t=jnp.zeros(self.m),
            fitness_elites_sorted=jnp.full((self.num_elites,), -jnp.inf),
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
        key_z, key_r = jax.random.split(key)
        z = jax.random.normal(key_z, (self.population_size, self.num_dims))
        r = jax.random.normal(key_r, (self.m,))

        # Compute std
        a = jnp.power(jnp.sqrt(1 - params.c_cov), jnp.arange(self.m + 1)[::-1])
        perturbation = a[0] * z + jnp.sqrt(params.c_cov) * jnp.dot(a[1:] * r, state.P)

        x = state.mean + state.std * perturbation
        return x, state

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
        elites_sorted = population[idx][: self.num_elites]
        fitness_elites_sorted = fitness[idx][: self.num_elites]

        # Update mean
        mean = state.mean + params.c_mean * jnp.dot(
            params.weights[: self.num_elites], elites_sorted - state.mean
        )

        # Rank-based Success Rule (RSR)
        F = jnp.concatenate([state.fitness_elites_sorted, fitness_elites_sorted])
        ranks = ranksort(F)
        q = (
            jnp.dot(
                params.weights[: self.num_elites],
                ranks[: self.num_elites] - ranks[self.num_elites :],
            )
            / self.num_elites
        )
        cumulative_rank_rate = (
            1 - params.c_std
        ) * state.cumulative_rank_rate + params.c_std * (q - params.q_star)

        # Update std
        std = state.std * jnp.exp(cumulative_rank_rate / params.d_std)
        std = jnp.clip(std, min=params.std_limit, max=1e8)  # Prevent extreme step sizes

        # Update evolution path
        p_c = (1 - params.c_c) * state.p_c + jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * (mean - state.mean) / state.std

        # Update P and t
        P_shifted = jnp.roll(state.P, shift=-1, axis=0)
        t_shifted = jnp.roll(state.t, shift=-1)

        # Compute condition
        if self.m > 1:
            generation_gap = state.t[1:] - state.t[:-1]
        else:
            generation_gap = jnp.zeros((self.m,))

        i_prime = jnp.argmin(generation_gap)
        T_min = generation_gap[i_prime]
        condition = (T_min > params.T) | (state.generation_counter < self.m)

        # Create masks for updating
        update_idx = jnp.arange(self.m)
        mask = update_idx >= i_prime

        # Update arrays using dynamic updates
        P = jnp.where(
            condition, P_shifted, jnp.where(mask[:, None], P_shifted, state.P)
        )
        t = jnp.where(condition, t_shifted, jnp.where(mask, t_shifted, state.t))

        # Final update
        P = P.at[-1].set(p_c)
        t = t.at[-1].set(state.generation_counter)

        return state.replace(
            mean=mean,
            std=std,
            p_c=p_c,
            cumulative_rank_rate=cumulative_rank_rate,
            P=P,
            t=t,
            fitness_elites_sorted=fitness_elites_sorted,
        )
