"""Separable Natural Evolution Strategy (Wierstra et al., 2014).

[1] https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import Params as BaseParams, State as BaseState, metrics_fn
from .xnes import xNES


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(BaseParams):
    std_init: float
    std_min: float
    std_max: float
    weights: jax.Array
    std_lr: float


class SNES(xNES):
    """Separable Natural Evolution Strategy (SNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SNES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(
            population_size, solution, optimizer, fitness_shaping_fn, metrics_fn
        )

    @property
    def _default_params(self) -> Params:
        params = super()._default_params

        # Override the learning rate for std with SNES-specific value
        std_lr = (3 + jnp.log(self.num_dims)) / (5 * jnp.sqrt(self.num_dims))

        return Params(
            std_init=params.std_init,
            std_min=0.0,
            std_max=1e8,
            weights=params.weights,
            std_lr=std_lr,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
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
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z = jnp.concatenate([z_plus, -z_plus])
        population = state.mean + state.std * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z = (population - state.mean) / state.std

        # Compute grad
        grad_mean = state.std * jnp.dot(fitness, z)

        # Update mean
        updates, opt_state = self.optimizer.update(-grad_mean, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Compute grad for std
        grad_std = jnp.dot(fitness, z**2 - 1)

        # Update std
        std = state.std * jnp.exp(0.5 * params.std_lr * grad_std)
        std = jnp.clip(std, min=params.std_min, max=params.std_max)

        return state.replace(mean=mean, std=std, opt_state=opt_state)
