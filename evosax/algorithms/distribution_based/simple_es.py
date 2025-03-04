"""Simple Evolution Strategy (Rechenberg, 1973).

Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
Inspired by: https://github.com/hardmaru/estool/blob/master/es.py
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState


@struct.dataclass
class Params(Params):
    std_init: float  # Standard deviation
    weights: jax.Array  # Weights for population members
    c_std: float  # Learning rate for population std


class SimpleES(DistributionBasedAlgorithm):
    """Simple Evolution Strategy (Simple ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize Simple ES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

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
        idx_sorted = jnp.argsort(fitness)
        y = population[idx_sorted] - state.mean

        # Compute grad
        grad = -jnp.dot(params.weights, y)

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Update std
        std_update = jnp.sqrt(jnp.dot(params.weights, y**2))
        std = (1 - params.c_std) * state.std + params.c_std * std_update

        return state.replace(mean=mean, std=std, opt_state=opt_state)
