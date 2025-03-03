"""Simple Evolution Strategy (Rechenberg, 1973).

Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    weights: jax.Array  # Weights for population members


@struct.dataclass
class Params(Params):
    std_init: float  # Standard deviation
    c_mean: float  # Learning rate for population mean
    c_std: float  # Learning rate for population std


class SimpleES(DistributionBasedAlgorithm):
    """Simple Evolution Strategy (Simple ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Simple ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            c_mean=1.0,
            c_std=0.1,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        weights = get_weights(self.population_size, self.elite_ratio)

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            weights=weights,
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
        idx_sorted = jnp.argsort(fitness)
        y = population[idx_sorted] - state.mean

        # Update mean
        mean_update = jnp.dot(state.weights, y)
        mean = state.mean + params.c_mean * mean_update

        # Update std
        std_update = jnp.sqrt(jnp.dot(state.weights, y**2))
        std = (1 - params.c_std) * state.std + params.c_std * std_update

        return state.replace(mean=mean, std=std)


def get_weights(population_size: int, elite_ratio: float):
    """Get weights for fitness shaping."""
    num_elites = jnp.asarray(elite_ratio * population_size, dtype=jnp.int32)
    mask = jnp.arange(population_size) < num_elites
    return mask * jnp.ones((population_size,)) / num_elites
