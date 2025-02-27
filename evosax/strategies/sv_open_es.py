"""Stein Variational OpenAI-ES (Liu et al., 2017).

Reference: https://arxiv.org/abs/1704.02399
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import metrics_fn
from evosax.core import exp_decay
from evosax.strategies.open_es import OpenES, Params, State

from ..types import Fitness, Population, Solution
from ..utils.kernel import kernel_rbf


@struct.dataclass
class Params(Params):
    kernel_std: float
    alpha: float


class SV_OpenES(OpenES):
    """Stein Variational OpenAI-ES (SV-OpenAI-ES)."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        use_antithetic_sampling: bool = True,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize SV-OpenAI-ES."""
        assert population_size % num_populations == 0, (
            "population_size must be divisible by num_populations."
        )
        super().__init__(
            population_size=population_size,
            solution=solution,
            use_antithetic_sampling=use_antithetic_sampling,
            opt_name=opt_name,
            lrate_init=lrate_init,
            lrate_decay=lrate_decay,
            lrate_limit=lrate_limit,
            metrics_fn=metrics_fn,
            **fitness_kwargs,
        )
        self.strategy_name = "SV_OpenES"

        self.num_populations = num_populations
        self.total_population_size = num_populations * population_size

        self.kernel = kernel

    @property
    def _default_params(self) -> Params:
        params = super()._default_params
        return Params(
            std_init=params.std_init,
            std_decay=params.std_decay,
            std_limit=params.std_limit,
            opt_params=params.opt_params,
            kernel_std=1.0,
            alpha=1.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        keys = jax.random.split(key, num=self.num_populations)
        state = jax.vmap(super()._init, in_axes=(0, None))(keys, params)
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

        # OpenAI-ES gradient
        grad = jax.vmap(jnp.dot)(
            fitness, (state.mean[:, None] - population) / state.std[:, None, None]
        ) / (self.population_size * state.std[:, None])

        # Compute SVGD steps
        svgd_grad = svgd_grad_fn(state.mean, grad, self.kernel, params)
        svgd_grad_kernel = svgd_grad_kernel_fn(state.mean, grad, self.kernel, params)
        grad = -(svgd_grad + params.alpha * svgd_grad_kernel)

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = jax.vmap(self.optimizer.step, (0, 0, 0, None))(
            state.mean, grad, state.opt_state, params.opt_params
        )
        opt_state = jax.vmap(self.optimizer.update, (0, None))(
            opt_state, params.opt_params
        )

        std = jax.vmap(exp_decay, (0, None, None))(
            state.std, params.std_decay, params.std_limit
        )
        return state.replace(mean=mean, std=std, opt_state=opt_state)


def svgd_grad_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD driving force."""

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: kernel(xj, xi, params) * gradj)(x, grad),
            axis=0,
        )

    return jax.vmap(phi)(x)


def svgd_grad_kernel_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD repulsive force."""

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: jax.grad(kernel)(xj, xi, params))(x, grad),
            axis=0,
        )

    return jax.vmap(phi)(x)
