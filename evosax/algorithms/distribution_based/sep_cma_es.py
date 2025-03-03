"""Separable CMA-ES (Ros & Hansen, 2008).

Reference: https://hal.inria.fr/inria-00287367/document
CMA-ES reference: https://arxiv.org/abs/1604.00772
Inspired by: github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_sepcma.py
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import State, metrics_fn
from .cma_es import CMA_ES, Params


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    p_std: jax.Array
    p_c: jax.Array
    C: jax.Array
    D: jax.Array


@struct.dataclass
class Params(Params):
    pass


class Sep_CMA_ES(CMA_ES):
    """Separable CMA-ES (Sep-CMA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Sep-CMA-ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5
        self.use_negative_weights = False

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            p_c=jnp.zeros(self.num_dims),
            C=jnp.ones(self.num_dims),
            D=jnp.ones(self.num_dims),
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
        # Compute D via eigen decomposition of C
        C, D = eigen_decomposition(state.C)

        # Sample new population
        z = jax.random.normal(key, (self.population_size, self.num_dims))  # Eq. (38)
        z = D * z  # Eq. (39)
        x = state.mean + state.std * z  # Eq. (40)

        return x, state.replace(C=C, D=D)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update mean
        mean, y_k, y_w = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # Cumulative Step length Adaptation (CSA)
        p_std = self.update_p_std(state.p_std, y_w / state.D, params)
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = self.update_std(state.std, norm_p_std, params)

        # Covariance matrix adaptation
        h_std = self.h_std(norm_p_std, state.generation_counter + 1, params)
        p_c = self.update_p_c(state.p_c, h_std, y_w, params)

        delta_h_std = self.delta_h_std(h_std, params)
        rank_one = self.rank_one(p_c)
        rank_mu = self.rank_mu(y_k, y_k / state.D, params)
        C = self.update_C(state.C, delta_h_std, rank_one, rank_mu, params)

        return state.replace(mean=mean, std=std, p_std=p_std, p_c=p_c, C=C)

    def rank_one(self, p_c: jax.Array) -> jax.Array:
        """Compute the rank-one update term for the covariance matrix."""
        return p_c**2

    def rank_mu(
        self, y_k: jax.Array, C_inv_sqrt_y_k: jax.Array, params: Params
    ) -> jax.Array:
        """Compute the rank-mu update term for the covariance matrix."""
        w_o = params.weights * jnp.where(
            params.weights >= 0,
            1,
            self.num_dims
            / jnp.clip(jnp.sum(jnp.square(C_inv_sqrt_y_k), axis=-1), min=1e-8),
        )  # Eq. (46)
        return jnp.dot(w_o, y_k**2)


def eigen_decomposition(C: jax.Array) -> jax.Array:
    """Eigendecomposition of covariance matrix."""
    C = jnp.clip(C, min=0.0, max=1e8)

    # Diagonal loading
    eps = 1e-8
    C = C + eps

    D = jnp.sqrt(C)
    return C, D
