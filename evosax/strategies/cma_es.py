"""Covariance Matrix Adaptation Evolution Strategy (Hansen et al., 2001).

Reference: https://arxiv.org/abs/1604.00772
Inspired by: https://github.com/CyberAgentAILab/cmaes
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
    p_c: jax.Array
    C: jax.Array
    B: jax.Array
    D: jax.Array
    grad: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    mu_eff: float
    c_mean: float
    c_std: float
    d_std: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float


class CMA_ES(Strategy):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize CMA-ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "CMA_ES"

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
            min_alpha * weights_prime / negative_sum,  # Zeroing out is also possible
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
            grad=0.0,
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
        # Sort
        idx = jnp.argsort(fitness)

        # Update mean
        y_k = (population[idx] - state.mean) / state.std  # ~ N(0, C)
        y_w = jnp.dot(
            params.weights[: self.num_elites], y_k[: self.num_elites]
        )  # Eq. (41)
        mean = state.mean + params.c_mean * (state.std * y_w + state.grad)  # Eq. (42)

        # Cumulative Step length Adaptation (CSA)
        C_inv_sqrt = state.B @ jnp.diag(1 / state.D) @ state.B.T
        p_std = (1 - params.c_std) * state.p_std + jnp.sqrt(
            params.c_std * (2 - params.c_std) * params.mu_eff
        ) * C_inv_sqrt @ y_w  # Eq. (43)
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = state.std * jnp.exp(
            (params.c_std / params.d_std) * (norm_p_std / params.chi_n - 1)
        )  # Eq. (44)

        # Covariance matrix adaptation
        h_std_cond_left = norm_p_std / jnp.sqrt(
            1 - (1 - params.c_std) ** (2 * (state.generation_counter + 1))
        )
        h_std_cond_right = (1.4 + 2 / (self.num_dims + 1)) * params.chi_n
        h_std = h_std_cond_left < h_std_cond_right  # Page 28
        p_c = (1 - params.c_c) * state.p_c + h_std * jnp.sqrt(
            params.c_c * (2 - params.c_c) * params.mu_eff
        ) * y_w  # Eq. (45)

        w_o = params.weights * jnp.where(
            params.weights >= 0,
            1,
            self.num_dims / (jnp.sum(jnp.square(y_k @ C_inv_sqrt.T), axis=-1) + 1e-8),
        )  # Eq. (46)
        delta_h_std = (1 - h_std) * params.c_c * (2 - params.c_c)  # Page 28
        rank_one = jnp.outer(p_c, p_c)
        rank_mu = jnp.einsum("i,ij,ik->jk", w_o, y_k, y_k)
        C = (
            (
                1
                + params.c_1 * delta_h_std
                - params.c_1
                - params.c_mu * jnp.sum(params.weights)
            )
            * state.C
            + params.c_1 * rank_one
            + params.c_mu * rank_mu
        )  # Eq. (47)

        return state.replace(mean=mean, std=std, p_std=p_std, p_c=p_c, C=C)


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
