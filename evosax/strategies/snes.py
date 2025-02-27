"""Separable Natural Evolution Strategy (Wierstra et al., 2014).

Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    weights: jax.Array
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    lrate_mean: float
    lrate_std: float


class SNES(Strategy):
    """Separable Natural Evolution Strategy (SNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize SNES."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "SNES"

    @property
    def _default_params(self) -> Params:
        lrate_std = (3 + jnp.log(self.num_dims)) / (5 * jnp.sqrt(self.num_dims))
        return Params(
            std_init=1.0,
            lrate_mean=1.0,
            lrate_std=lrate_std,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        weights = get_weights(self.population_size)

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            weights=weights,
            z=jnp.zeros((self.population_size, self.num_dims)),
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
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z_sorted = state.z[fitness.argsort()]
        weights = jnp.expand_dims(state.weights, axis=-1)

        # Update mean
        grad_mean = jnp.sum(weights * z_sorted, axis=0)
        mean = state.mean + params.lrate_mean * state.std * grad_mean

        # Update std
        grad_std = jnp.sum(weights * (z_sorted**2 - 1), axis=0)
        std = state.std * jnp.exp(params.lrate_std / 2 * grad_std)

        return state.replace(mean=mean, std=std)


def get_weights(population_size: int):
    """Get weights for fitness shaping."""
    weights = jnp.clip(
        jnp.log(population_size / 2 + 1) - jnp.log(jnp.arange(1, population_size + 1)),
        min=0.0,
    )
    weights = weights / jnp.sum(weights)
    return weights - 1 / population_size
