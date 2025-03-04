"""Discovered Evolution Strategy (Lange et al., 2023)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    temperature: float  # Temperature for softmax weights
    lr_mean: float  # Learning rate for population mean
    lr_std: float  # Learning rate for population std


class DES(DistributionBasedAlgorithm):
    """Discovered Evolution Strategy (DES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize DES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

    @property
    def _default_params(self) -> Params:
        temperature = 12.5
        weights = get_weights(self.population_size, temperature)
        return Params(
            std_init=1.0,
            weights=weights,
            temperature=temperature,
            lr_mean=1.0,
            lr_std=0.1,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
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
        population = state.mean + state.std[None, ...] * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        x = population[fitness.argsort()]

        # Weighted updates
        weighted_mean = jnp.dot(params.weights, x)
        weighted_std = jnp.sqrt(jnp.dot(params.weights, (x - state.mean) ** 2))

        mean = state.mean + params.lr_mean * (weighted_mean - state.mean)
        std = state.std + params.lr_std * (weighted_std - state.std)
        return state.replace(mean=mean, std=std)


def get_weights(population_size: int, temperature: float = 12.5):
    """Get weights for fitness shaping."""
    ranks = jnp.arange(population_size)
    sigmoid = nn.sigmoid(temperature * (ranks / population_size - 0.5))
    return nn.softmax(20 * sigmoid)
