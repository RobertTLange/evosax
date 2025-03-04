"""Gradientless Descent (Golovin et al., 2019).

Reference: https://arxiv.org/abs/1911.06317
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: Solution


@struct.dataclass
class Params(Params):
    radius_min: float
    radius_max: float
    radius_decay: float


class GradientlessDescent(DistributionBasedAlgorithm):
    """GradientLess Descent (GLD)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize GLD."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

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
        return state.replace(mean=state.best_solution)
