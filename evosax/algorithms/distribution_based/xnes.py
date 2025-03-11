"""Exponential Natural Evolution Strategy (Wierstra et al., 2014).

[1] https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
[2] https://github.com/chanshing/xnes
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
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
    opt_state: optax.OptState
    B: jax.Array
    lr_std: float
    z: jax.Array


@struct.dataclass
class Params(BaseParams):
    std_init: float
    weights: jax.Array
    lr_std_init: float
    lr_B: float


class xNES(DistributionBasedAlgorithm):
    """Exponential Natural Evolution Strategy (xNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize xNES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

    @property
    def _default_params(self) -> Params:
        weights = get_weights(self.population_size)

        lr_std_init = (9 + 3 * jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims) * self.num_dims
        )
        return Params(
            std_init=1.0,
            weights=weights,
            lr_std_init=lr_std_init,
            lr_B=lr_std_init,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            B=params.std_init * jnp.eye(self.num_dims),
            lr_std=params.lr_std_init,
            z=jnp.zeros((self.population_size, self.num_dims)),
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
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z @ state.B
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Compute grad for mean
        grad_mean = -state.std * state.B @ jnp.dot(fitness, state.z)

        # Update mean
        updates, opt_state = self.optimizer.update(grad_mean, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Compute grad for std
        grad_M = jnp.einsum(
            "i,ijk->jk",
            fitness,
            jax.vmap(jnp.outer)(state.z, state.z) - jnp.eye(self.num_dims),
        )
        grad_std = jnp.trace(grad_M) / self.num_dims

        # Update std
        std = state.std * jnp.exp(0.5 * state.lr_std * grad_std)

        # Update B
        grad_B = grad_M - grad_std * jnp.eye(self.num_dims)
        B = state.B * jnp.exp(0.5 * params.lr_B * grad_B)

        return state.replace(mean=mean, std=std, opt_state=opt_state, B=B)


def get_weights(population_size: int):
    """Get weights for fitness shaping."""
    weights = jnp.clip(
        jnp.log(population_size / 2 + 1) - jnp.log(jnp.arange(1, population_size + 1)),
        min=0.0,
    )
    weights = weights / jnp.sum(weights)
    return weights - 1 / population_size
