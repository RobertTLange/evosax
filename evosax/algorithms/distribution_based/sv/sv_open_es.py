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

from ..open_es import Open_ES, Params as BaseParams, State as BaseState
from .base import SV_ES, metrics_fn


@struct.dataclass
class State(BaseState):
    pass


@struct.dataclass
class Params(BaseParams):
    kernel_std: float
    alpha: float


class SV_Open_ES(SV_ES, Open_ES):
    """Stein Variational OpenAI-ES (SV-OpenAI-ES).

    Runs `num_populations` OpenAI-ES search distributions in parallel. The
    gradient estimates of all distributions are combined with Stein variational
    gradient descent: a kernel-weighted average of the gradients plus the kernel
    gradient between the means, scaled by `params.alpha`.

    Args:
        population_size: Number of candidate solutions sampled per search
            distribution in each generation. Must be even.
        num_populations: Number of search distributions run in parallel.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        kernel: Function `(x, y, params) -> similarity` between the means of
            two search distributions. See `evosax.core.kernel`.
        use_antithetic_sampling: If True, sample perturbations in mirrored pairs
            `(z, -z)`.
        optimizer: Optax optimizer applied per search distribution to the Stein
            variational gradient to update its mean.
        std_schedule: Function mapping the generation counter to the standard
            deviation of the search distribution, e.g. an Optax schedule.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied
            per search distribution to the raw fitness in `tell` before the
            algorithm update. See `evosax.core.fitness_shaping`.
        metrics_fn: Function
            `(key, population, fitness, state, params) -> metrics` evaluated
            per search distribution in `tell`; its output is returned
            alongside the new state.

    References:
        - [Liu et al. (2017)](https://arxiv.org/abs/1704.02399)
    """

    def __init__(
        self,
        population_size: int,
        num_populations: int,
        solution: Solution,
        kernel: Callable = kernel_rbf,
        use_antithetic_sampling: bool = True,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
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
            std_schedule,
            fitness_shaping_fn,
            metrics_fn,
        )

    @property
    def _default_params(self) -> Params:
        return Params(kernel_std=1.0, alpha=1.0)

    def _tell(  # type: ignore[override]
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
        updates, opt_state = jax.vmap(self.optimizer.update)(
            grad, state.opt_state, state.mean
        )
        mean = jax.vmap(optax.apply_updates)(state.mean, updates)

        return state.replace(
            mean=mean,
            std=jax.vmap(self.std_schedule)(state.generation_counter),
            opt_state=opt_state,
        )


def svgd_grad_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD driving force.

    Args:
        x: Means of all search distributions, shape `(num_populations, num_dims)`.
        grad: Gradient estimate of each search distribution, same shape as `x`.
        kernel: Function `(x, y, params) -> similarity` between two means.
        params: Algorithm hyperparameters, passed to `kernel`.

    Returns:
        The kernel-weighted average of the gradients for each mean, same shape as
        `x`.
    """

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: gradj * kernel(xj, xi, params))(x, grad),
            axis=0,
        )  # Eq. (10)

    return jax.vmap(phi)(x)


def svgd_grad_kernel_fn(
    x: jax.Array, grad: jax.Array, kernel: Callable, params: Params
) -> jax.Array:
    """SVGD repulsive force.

    Args:
        x: Means of all search distributions, shape `(num_populations, num_dims)`.
        grad: Gradient estimate of each search distribution. Not used by this
            term.
        kernel: Function `(x, y, params) -> similarity` between two means.
        params: Algorithm hyperparameters, passed to `kernel`.

    Returns:
        The average kernel gradient for each mean, same shape as `x`.
    """

    def phi(xi):
        return jnp.mean(
            jax.vmap(lambda xj, gradj: jax.grad(kernel)(xj, xi, params))(x, grad),
            axis=0,
        )  # Eq. (10)

    return jax.vmap(phi)(x)
