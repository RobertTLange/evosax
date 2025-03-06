"""Stein Variational CMA-ES (Braun et al., 2024).

[1] https://arxiv.org/abs/2410.10390
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.core.kernel import kernel_rbf
from evosax.types import Fitness, Population, Solution

from ..cma_es import CMA_ES, Params, State
from .base import SV_ES, metrics_fn


@struct.dataclass
class State(State):
    pass


@struct.dataclass
class Params(Params):
    kernel_std: float
    alpha: float


class SV_CMA_ES(SV_ES, CMA_ES):
    """Stein Variational CMA-ES (SV-CMA-ES)."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SV-CMA-ES."""
        SV_ES.__init__(
            self,
            population_size,
            num_populations,
            solution,
            kernel,
            fitness_shaping_fn,
            metrics_fn,
        )

        CMA_ES.__init__(
            self,
            population_size,
            solution,
            fitness_shaping_fn,
            metrics_fn,
        )

    @property
    def _default_params(self) -> Params:
        params = super()._default_params
        return Params(
            std_init=params.std_init,
            std_min=params.std_min,
            std_max=params.std_max,
            weights=params.weights,
            mu_eff=params.mu_eff,
            c_mean=params.c_mean,
            c_std=params.c_std,
            d_std=params.d_std,
            c_c=params.c_c,
            c_1=params.c_1,
            c_mu=params.c_mu,
            chi_n=params.chi_n,
            kernel_std=1.0,
            alpha=1.0,
        )

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update mean with SV gradient
        mean, y_k, y_w = jax.vmap(self.update_mean, in_axes=(0, 0, 0, 0, None, None))(
            population, fitness, state.mean, state.std, state.mean, params
        )

        # Update p_std
        C_inv_sqrt_y_w = jax.vmap(lambda B, D, y: B @ ((B.T @ y) / D))(
            state.B, state.D, y_w
        )
        p_std = jax.vmap(self.update_p_std, in_axes=(0, 0, None))(
            state.p_std, C_inv_sqrt_y_w, params
        )
        norm_p_std = jax.vmap(jnp.linalg.norm)(p_std)

        # Update std
        std = jax.vmap(self.update_std, in_axes=(0, 0, None))(
            state.std, norm_p_std, params
        )

        # Covariance matrix adaptation
        h_std = jax.vmap(self.h_std, in_axes=(0, 0, None))(
            norm_p_std, state.generation_counter + 1, params
        )
        p_c = jax.vmap(self.update_p_c, in_axes=(0, 0, 0, None))(
            state.p_c, h_std, y_w, params
        )

        delta_h_std = jax.vmap(self.delta_h_std, in_axes=(0, None))(h_std, params)
        rank_one = jax.vmap(self.rank_one)(p_c)

        # Compute rank_mu updates using the original y_k
        C_inv_sqrt_y_k = jax.vmap(lambda y, B, D: (y @ B) * (1 / D) @ B.T)(
            y_k, state.B, state.D
        )
        rank_mu = jax.vmap(self.rank_mu)(fitness, y_k, C_inv_sqrt_y_k)

        C = jax.vmap(self.update_C, in_axes=(0, 0, 0, 0, None))(
            state.C, delta_h_std, rank_one, rank_mu, params
        )

        return state.replace(
            mean=mean,
            std=std,
            p_std=p_std,
            p_c=p_c,
            C=C,
        )

    def update_mean(
        self,
        population: Population,
        fitness: Fitness,
        mean: jax.Array,
        std: float,
        means: Solution,
        params: Params,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Update the mean of the distribution with Stein variational gradient."""
        y_k = (population - mean) / std  # ~ N(0, C)
        y_w = jnp.dot(jnp.where(fitness < 0.0, 0.0, fitness), y_k)  # Eq. (41)

        # Compute kernel gradient
        grad_kernel = jnp.mean(
            jax.vmap(lambda xj: jax.grad(self.kernel)(xj, mean, params))(
                means,
            ),
            axis=0,
        )

        # Apply Stein variational gradient
        y_w += params.alpha * grad_kernel / std

        # Update mean
        return mean + params.c_mean * std * y_w, y_k, y_w
