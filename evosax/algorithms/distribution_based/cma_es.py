"""Covariance Matrix Adaptation Evolution Strategy (Hansen et al., 2001).

[1] https://arxiv.org/abs/1604.00772
[2] https://github.com/CyberAgentAILab/cmaes
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import (
    DistributionBasedAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn,
)


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: float
    p_std: jax.Array
    p_c: jax.Array
    C: jax.Array
    B: jax.Array
    D: jax.Array


@struct.dataclass
class Params(BaseParams):
    std_init: float
    std_min: float
    std_max: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float


class CMA_ES(DistributionBasedAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize CMA-ES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        self.use_negative_weights = True

    @property
    def _default_params(self) -> Params:
        weights_prime = jnp.log((self.population_size + 1) / 2) - jnp.log(
            jnp.arange(1, self.population_size + 1)
        )  # Eq. (48)

        mu_eff = jnp.sum(weights_prime[: self.num_elites]) ** 2 / jnp.sum(
            weights_prime[: self.num_elites] ** 2
        )  # Eq. (8)
        mu_eff_minus = jnp.sum(weights_prime[self.num_elites :]) ** 2 / jnp.sum(
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
            self.use_negative_weights * min_alpha * weights_prime / negative_sum,
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
            std_min=0.0,
            std_max=1e8,
            weights=weights,
            mu_eff=mu_eff,
            c_mean=c_mean,
            c_std=c_std,
            d_std=d_std,
            c_c=c_c,
            c_1=c_1,
            c_mu=c_mu,
            chi_n=chi_n,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            p_c=jnp.zeros(self.num_dims),
            C=jnp.eye(self.num_dims),
            B=jnp.eye(self.num_dims),
            D=jnp.ones((self.num_dims,)),
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
        # Compute B and D via eigen decomposition of C
        C, B, D = eigen_decomposition(state.C)

        # Sample new population
        z = jax.random.normal(key, (self.population_size, self.num_dims))  # Eq. (38)
        z = (z @ jnp.diag(D).T) @ B.T  # Eq. (39)
        population = state.mean + state.std * z  # Eq. (40)

        return population, state.replace(C=C, B=B, D=D)

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
        p_std = self.update_p_std(
            state.p_std, state.B @ ((state.B.T @ y_w) / state.D), params
        )
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = self.update_std(state.std, norm_p_std, params)

        # Covariance matrix adaptation
        h_std = self.h_std(norm_p_std, state.generation_counter + 1, params)
        p_c = self.update_p_c(state.p_c, h_std, y_w, params)

        delta_h_std = self.delta_h_std(h_std, params)
        rank_one = self.rank_one(p_c)
        rank_mu = self.rank_mu(
            fitness, y_k, (y_k @ state.B) * (1 / state.D) @ state.B.T
        )
        C = self.update_C(state.C, delta_h_std, rank_one, rank_mu, params)

        return state.replace(mean=mean, std=std, p_std=p_std, p_c=p_c, C=C)

    def update_mean(
        self,
        population: Population,
        fitness: Fitness,
        mean: jax.Array,
        std: float,
        params: Params,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Update the mean of the distribution."""
        y_k = (population - mean) / std  # ~ N(0, C)
        y_w = jnp.dot(jnp.where(fitness < 0.0, 0.0, fitness), y_k)  # Eq. (41)
        return mean + params.c_mean * std * y_w, y_k, y_w  # Eq. (42)

    def update_p_std(
        self, p_std: jax.Array, C_inv_sqrt_y_w: jax.Array, params: Params
    ) -> jax.Array:
        """Update the evolution path for the step-size adaptation."""
        return (1 - params.c_std) * p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * C_inv_sqrt_y_w  # Eq. (43)

    def update_std(self, std: float, norm_p_std: float, params: Params) -> float:
        """Update the step size (standard deviation)."""
        std = std * jnp.exp(
            (params.c_std / params.d_std) * (norm_p_std / params.chi_n - 1)
        )  # Eq. (44)
        return jnp.clip(std, min=params.std_min, max=params.std_max)

    def h_std(self, norm_p_std: float, generation_counter: int, params: Params) -> bool:
        """Compute the stall indicator for the rank-one update."""
        h_std_cond_left = norm_p_std / jnp.sqrt(
            1 - (1 - params.c_std) ** (2 * (generation_counter + 1))
        )
        h_std_cond_right = (1.4 + 2 / (self.num_dims + 1)) * params.chi_n
        return h_std_cond_left < h_std_cond_right  # Page 28

    def update_p_c(
        self, p_c: jax.Array, h_std: bool, y_w: jax.Array, params: Params
    ) -> jax.Array:
        """Update the evolution path for the covariance matrix adaptation."""
        return (1 - params.c_c) * p_c + h_std * jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * y_w  # Eq. (45)

    def delta_h_std(self, h_std: bool, params: Params) -> float:
        """Compute the coefficient for the rank-one update when stalled."""
        return (1 - h_std) * params.c_c * (2 - params.c_c)  # Page 28

    def rank_one(self, p_c: jax.Array) -> jax.Array:
        """Compute the rank-one update term for the covariance matrix."""
        return jnp.outer(p_c, p_c)

    def rank_mu(
        self, fitness: Fitness, y_k: jax.Array, C_inv_sqrt_y_k: jax.Array
    ) -> jax.Array:
        """Compute the rank-mu update term for the covariance matrix."""
        w_o = fitness * jnp.where(
            fitness >= 0,
            1,
            self.num_dims
            / jnp.clip(jnp.sum(jnp.square(C_inv_sqrt_y_k), axis=-1), min=1e-8),
        )  # Eq. (46)
        return jnp.einsum("i,ij,ik->jk", w_o, y_k, y_k)

    def update_C(
        self,
        C: jax.Array,
        delta_h_std: float,
        rank_one: jax.Array,
        rank_mu: jax.Array,
        params: Params,
    ) -> jax.Array:
        """Update the covariance matrix."""
        return (
            (
                1
                + params.c_1 * delta_h_std
                - params.c_1
                - params.c_mu * jnp.sum(params.weights)
            )
            * C
            + params.c_1 * rank_one
            + params.c_mu * rank_mu
        )  # Eq. (47)


def eigen_decomposition(C: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Eigendecomposition of covariance matrix."""
    # Symmetry
    C = (C + C.T) / 2

    # Clip diagonal elements to 0 and all elements to max 1e8
    diag_indices = jnp.diag_indices_from(C)
    C = C.at[diag_indices].set(jnp.maximum(C[diag_indices], 0.0))
    C = jnp.minimum(C, 1e8)

    # Diagonal loading
    eps = 1e-8
    C = C + eps * jnp.eye(C.shape[0])

    # Compute eigen decomposition
    D_sq, B = jnp.linalg.eigh(C)

    D = jnp.sqrt(jnp.maximum(D_sq, eps))

    return C, B, D
