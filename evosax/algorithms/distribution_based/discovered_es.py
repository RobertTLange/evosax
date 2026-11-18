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
    """Discovered Evolution Strategy (DES).

    Samples the population from a Gaussian with a per-dimension standard
    deviation. Population members are weighted by rank with the weights from
    `get_weights`. The mean is updated with an Optax optimizer towards the
    weighted average of the population, and the standard deviation moves towards
    the weighted root-mean-square deviation from the mean with learning rate
    `params.lr_std`.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        optimizer: Optax optimizer applied to the estimated gradient to update the
            mean.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`. The update uses the shaped fitness as
            recombination weights; the default assigns `params.weights` by
            fitness rank.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260)
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
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
        updates, opt_state = self.optimizer.update(
            grad_mean, state.opt_state, state.mean
        )
        mean = optax.apply_updates(state.mean, updates)

        # Update std
        std = state.std - params.lr_std * grad_std

        return state.replace(mean=mean, std=std, opt_state=opt_state)


def get_weights(population_size: int, temperature: float = 12.5):
    """Compute the rank-based recombination weights of DES.

    The weights are `softmax(-20 * sigmoid(temperature * centered_ranks))` with
    `centered_ranks` evenly spaced in `[-0.5, 0.5]`.

    Args:
        population_size: Number of weights to compute.
        temperature: Scale applied to the centered ranks inside the sigmoid.

    Returns:
        Weights of shape `(population_size,)` that sum to 1, ordered from the best
        rank (largest weight) to the worst.
    """
    centered_ranks = jnp.arange(population_size) / (population_size - 1) - 0.5
    sigmoid = nn.sigmoid(temperature * centered_ranks)
    return nn.softmax(-20 * sigmoid)
