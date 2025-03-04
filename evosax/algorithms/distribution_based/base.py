"""Base module for distribution-based algorithms."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Metrics, Population, Solution
from ..base import EvolutionaryAlgorithm, Params, State
from ..base import metrics_fn as base_metrics_fn


@struct.dataclass
class State(State):
    mean: Solution


@struct.dataclass
class Params(Params):
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
    return metrics | {"mean": state.mean, "mean_norm": jnp.linalg.norm(state.mean)}


class DistributionBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for distribution-based algorithms."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize base class for distribution-based algorithm."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

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
