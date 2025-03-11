"""Base module for distribution-based algorithms."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Metrics, Population, Solution

from ..base import (
    EvolutionaryAlgorithm,
    Params as BaseParams,
    State as BaseState,
    metrics_fn as base_metrics_fn,
)


@struct.dataclass
class State(BaseState):
    mean: Solution


@struct.dataclass
class Params(BaseParams):
    pass


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Compute metrics for distribution-based algorithm."""
    metrics = base_metrics_fn(key, population, fitness, state, params)
    return metrics | {
        "mean": state.mean,
        "mean_norm": jnp.linalg.norm(state.mean, axis=-1),
    }


class DistributionBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for distribution-based algorithms."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize base class for distribution-based algorithm."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        mean: Solution,
        params: Params,
    ) -> State:
        """Initialize distribution-based algorithm."""
        state = self._init(key, params)
        state = state.replace(mean=self._ravel_solution(mean))
        return state

    def get_mean(self, state: State) -> Solution:
        """Return unravelled mean."""
        mean = self._unravel_solution(state.mean)
        return mean
