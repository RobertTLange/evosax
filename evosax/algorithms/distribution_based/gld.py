"""Gradientless Descent (Golovin et al., 2019).

Reference: https://arxiv.org/abs/1911.06317
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: Solution
    fitness: float


@struct.dataclass
class Params(Params):
    radius_min: float
    radius_max: float
    radius_decay: float


class GLD(DistributionBasedAlgorithm):
    """Gradientless Descent (GLD)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize GLD."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

    @property
    def _default_params(self) -> Params:
        return Params(
            radius_min=0.001,
            radius_max=0.2,
            radius_decay=5.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            fitness=jnp.inf,
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

        # Exponentially decaying radius
        radius = (
            params.radius_min
            + jnp.exp2(  # TODO: different from the original paper
                -jnp.arange(self.population_size) / params.radius_decay
            )
            * (params.radius_max - params.radius_min)
        )

        x = state.mean + radius[..., None] * z
        return x, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Get best member in current population
        best_idx = jnp.argmin(fitness)
        best_member, best_fitness = population[best_idx], fitness[best_idx]

        # Replace if new best
        replace = best_fitness < state.fitness
        best_member = jnp.where(replace, best_member, state.mean)
        best_fitness = jnp.where(replace, best_fitness, state.fitness)

        return state.replace(mean=best_member, fitness=best_fitness)
