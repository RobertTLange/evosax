"""Stein Variational CMA-ES (Braun et al., 2024).

Reference: https://arxiv.org/abs/2410.10390
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.strategies.cma_es import CMA_ES, Params, State

from ..strategy import metrics_fn
from ..types import Fitness, Population, Solution
from ..utils.kernel import kernel_rbf


@struct.dataclass
class Params(Params):
    kernel_std: float
    alpha: float


class SV_CMA_ES(CMA_ES):
    """Stein Variational CMA-ES (SV-CMA-ES)."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize SV-CMA-ES."""
        assert population_size % num_populations == 0, (
            "population_size must be divisible by num_populations."
        )
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.num_populations = num_populations
        self.total_population_size = num_populations * population_size

        self.kernel = kernel

    @property
    def _default_params(self) -> Params:
        params = super()._default_params
        return Params(
            std_init=params.std_init,
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

    def _init(self, key: jax.Array, params: Params) -> State:
        keys = jax.random.split(key, num=self.num_populations)
        state = jax.vmap(super()._init, in_axes=(0, None))(keys, params)
        state = state.replace(grad=jnp.zeros((self.num_populations, self.num_dims)))
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        keys = jax.random.split(key, num=self.num_populations)
        population, state = jax.vmap(super()._ask, in_axes=(0, 0, None))(
            keys, state, params
        )
        population = population.reshape(self.total_population_size, self.num_dims)
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        population = population.reshape(
            self.num_populations, self.population_size, self.num_dims
        )
        fitness = fitness.reshape(self.num_populations, self.population_size)

        # Compute kernel grads
        grad_kernel = jax.vmap(
            lambda xi: jnp.mean(
                jax.vmap(lambda xj: jax.grad(self.kernel)(xj, xi, params))(state.mean),
                axis=0,
            )
        )(state.mean)
        state = state.replace(grad=params.alpha * grad_kernel)

        keys = jax.random.split(key, num=self.num_populations)
        state = jax.vmap(super()._tell, in_axes=(0, 0, 0, 0, None))(
            keys, population, fitness, state, params
        )
        return state
