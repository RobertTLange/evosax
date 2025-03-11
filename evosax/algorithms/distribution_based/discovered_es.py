"""Discovered Evolution Strategy (Lange et al., 2023)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn, struct

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
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(BaseParams):
    std_init: float
    weights: jax.Array
    lr_std: float  # Learning rate for population std


class DiscoveredES(DistributionBasedAlgorithm):
    """Discovered Evolution Strategy (DES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize DES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

        self.temperature = 12.5

    @property
    def _default_params(self) -> Params:
        weights = get_weights(self.population_size, self.temperature)

        return Params(
            std_init=1.0,
            weights=weights,
            lr_std=0.1,
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
        z = jax.random.normal(key, (self.population_size, self.num_dims))
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
        # Compute grad
        grad_mean = -jnp.dot(fitness, population - state.mean)
        grad_std = -(
            jnp.sqrt(jnp.dot(fitness, (population - state.mean) ** 2)) - state.std
        )

        # Update mean
        updates, opt_state = self.optimizer.update(grad_mean, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Update std
        std = state.std - params.lr_std * grad_std

        return state.replace(mean=mean, std=std, opt_state=opt_state)


def get_weights(population_size: int, temperature: float = 12.5):
    """Get weights for fitness shaping."""
    centered_ranks = jnp.arange(population_size) / (population_size - 1) - 0.5
    sigmoid = nn.sigmoid(temperature * centered_ranks)
    return nn.softmax(-20 * sigmoid)
