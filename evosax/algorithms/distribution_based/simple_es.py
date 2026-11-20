"""Simple Evolution Strategy (Rechenberg, 1973).

[1] https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
[2] https://github.com/hardmaru/estool/blob/master/es.py
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
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(BaseParams):
    std_init: float
    weights: jax.Array
    c_std: float  # Learning rate for population std


class SimpleES(DistributionBasedAlgorithm):
    """Simple Evolution Strategy (Simple ES).

    Samples the population from a diagonal Gaussian. The mean is moved with an
    Optax optimizer towards the weighted average of the population, where the
    default weights are uniform over the better half. The standard deviation
    is an exponential moving average, with rate `params.c_std`, of the weighted
    spread of the population around the mean.

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
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Rechenberg (1973)](https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506)
        - [Reference implementation](https://github.com/hardmaru/estool/blob/master/es.py)
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

        self.elite_ratio = 0.5

        # Optimizer
        self.optimizer = optimizer

    @property
    def _default_params(self) -> Params:
        mask = jnp.arange(self.population_size) < self.num_elites
        weights = mask * jnp.ones((self.population_size,)) / self.num_elites

        return Params(
            std_init=1.0,
            weights=weights,
            c_std=0.1,
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
        grad = -jnp.dot(fitness, population - state.mean)

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.mean)
        mean = optax.apply_updates(state.mean, updates)

        # Update std
        std_update = jnp.sqrt(jnp.dot(fitness, (population - state.mean) ** 2))
        std = (1 - params.c_std) * state.std + params.c_std * std_update

        return state.replace(mean=mean, std=std, opt_state=opt_state)
