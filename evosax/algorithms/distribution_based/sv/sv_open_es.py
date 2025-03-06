"""Stein Variational OpenAI-ES (Liu et al., 2017).

[1] https://arxiv.org/abs/1704.02399
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn
from evosax.core.kernel import kernel_rbf
from evosax.types import Fitness, Population, Solution

from ..open_es import Open_ES, Params, State
from .base import SV_ES, metrics_fn


@struct.dataclass
class State(State):
    pass


@struct.dataclass
class Params(Params):
    kernel_std: float
    alpha: float


class SV_Open_ES(SV_ES, Open_ES):
    """Stein Variational OpenAI-ES (SV-OpenAI-ES)."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SV-OpenAI-ES."""
        SV_ES.__init__(
            self,
            population_size,
            num_populations,
            solution,
            kernel,
            fitness_shaping_fn,
            metrics_fn,
        )

        Open_ES.__init__(
            self,
            population_size,
            solution,
            use_antithetic_sampling,
            optimizer,
            fitness_shaping_fn,
            metrics_fn,
        )

    @property
    def _default_params(self) -> Params:
        params = super()._default_params
        return Params(
            std_init=params.std_init,
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
        # OpenAI-ES gradient
        grad = jax.vmap(jnp.dot)(
            fitness, (state.mean[:, None] - population) / state.std[:, None, None]
        ) / (self.population_size * state.std[:, None])

        # Compute SVGD steps
        svgd_grad = svgd_grad_fn(state.mean, grad, self.kernel, params)
        svgd_grad_kernel = svgd_grad_kernel_fn(state.mean, grad, self.kernel, params)
        grad = -(svgd_grad + params.alpha * svgd_grad_kernel)

        # Update mean
        updates, opt_state = jax.vmap(self.optimizer.update)(grad, state.opt_state)
        mean = jax.vmap(optax.apply_updates)(state.mean, updates)

        return state.replace(mean=mean, opt_state=opt_state)


def svgd_grad_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD driving force."""

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: gradj * kernel(xj, xi, params))(x, grad),
            axis=0,
        )  # Eq. (10)

    return jax.vmap(phi)(x)


def svgd_grad_kernel_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD repulsive force."""

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: jax.grad(kernel)(xj, xi, params))(x, grad),
            axis=0,
        )  # Eq. (10)

    return jax.vmap(phi)(x)
