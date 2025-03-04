"""Stein Variational OpenAI-ES (Liu et al., 2017).

Reference: https://arxiv.org/abs/1704.02399
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...core.kernel import kernel_rbf
from ...types import Fitness, Population, Solution
from .base import metrics_fn
from .open_es import Open_ES, Params, State


@struct.dataclass
class State(State):
    pass


@struct.dataclass
class Params(Params):
    kernel_std: float
    alpha: float


class SV_Open_ES(Open_ES):
    """Stein Variational OpenAI-ES (SV-OpenAI-ES)."""

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SV-OpenAI-ES."""
        super().__init__(
            population_size=population_size,
            solution=solution,
            use_antithetic_sampling=use_antithetic_sampling,
            optimizer=optimizer,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )

        self.num_populations = num_populations
        self.total_population_size = num_populations * population_size

        self.kernel = kernel

    @property
    def _default_params(self) -> Params:
        params = super()._default_params
        return Params(
            std_init=params.std_init,
            kernel_std=1.0,
            alpha=1.0,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        means: Solution,
        params: Params,
    ) -> State:
        """Initialize distribution-based algorithm."""
        state = self._init(key, params)

        state = state.replace(mean=jax.vmap(self._ravel_solution)(means))
        return state

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

        # Update mean
        updates, opt_state = jax.vmap(self.optimizer.update)(grad, state.opt_state)
        mean = jax.vmap(optax.apply_updates)(state.mean, updates)

        return state.replace(mean=mean, opt_state=opt_state)

    def get_mean(self, state: State) -> Solution:
        """Return unravelled mean."""
        mean = jax.vmap(self._unravel_solution)(state.mean)
        return mean


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
