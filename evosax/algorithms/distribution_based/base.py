"""Base module for distribution-based algorithms.

Distribution-based algorithms maintain a search distribution over solutions,
sample a population from it in `ask` and update the distribution in `tell`.
"""

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
    """State shared by all distribution-based algorithms.

    Attributes:
        mean: Flat mean of the search distribution.
    """

    mean: Solution


@struct.dataclass
class Params(BaseParams):
    """Hyperparameters shared by all distribution-based algorithms."""

    pass


def metrics_fn(
    key: jax.Array,
    population: Population,
    fitness: Fitness,
    state: State,
    params: Params,
) -> Metrics:
    """Compute the default metrics of distribution-based algorithms.

    Extends the base metrics with the mean of the search distribution and its
    norm. See `evosax.algorithms.base.metrics_fn` for the arguments.
    """
    metrics = base_metrics_fn(key, population, fitness, state, params)
    return metrics | {
        "mean": state.mean,
        "mean_norm": jnp.linalg.norm(state.mean, axis=-1),
    }


class DistributionBasedAlgorithm(EvolutionaryAlgorithm):
    """Base class for distribution-based algorithms.

    A distribution-based algorithm maintains a search distribution, typically a
    Gaussian with mean `state.mean`. `ask` samples a population from it and
    `tell` updates it from the fitness of that population. The state is
    initialized from an initial mean.

    Args:
        population_size: Number of candidate solutions sampled and evaluated
            per generation.
        solution: Example solution PyTree. Only its structure and leaf shapes
            are used: they set the number of search dimensions and how flat
            search vectors are unravelled into the PyTrees returned by `ask`.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied
            to the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function
            `(key, population, fitness, state, params) -> metrics` evaluated
            in `tell`; its output is returned alongside the new state.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        key: jax.Array,
        mean: Solution,
        params: Params,
    ) -> State:
        """Initialize the algorithm state from an initial mean.

        Args:
            key: PRNG key.
            mean: Initial mean of the search distribution, a PyTree with the
                structure of `solution`.
            params: Algorithm hyperparameters, e.g. `default_params`.

        Returns:
            The initial algorithm state.
        """
        state = self._init(key, params)
        state = state.replace(mean=self._ravel_solution(mean))
        return state

    def get_mean(self, state: State) -> Solution:
        """Return the mean of the search distribution as a solution PyTree."""
        mean = self._unravel_solution(state.mean)
        return mean
