"""Matrix Adaptation Evolution Strategy (Bayer & Sendhoff, 2017).

Reference: https://ieeexplore.ieee.org/document/7875115
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...types import Fitness, Population, Solution
from .base import State, metrics_fn
from .cma_es import CMA_ES, Params


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    p_std: jax.Array
    M: jax.Array
    z: jax.Array


@struct.dataclass
class Params(Params):
    pass


class MA_ES(CMA_ES):
    """Matrix Adaptation Evolution Strategy (MA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize MA-ES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        self.use_negative_weights = False

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            M=jnp.eye(self.num_dims),
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
        # Sample new population
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z @ state.M.T

        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Sort
        idx = jnp.argsort(fitness)
        state = state.replace(z=state.z[idx])

        # Update mean
        mean, _, _ = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # Cumulative Step length Adaptation (CSA)
        p_std = self.update_p_std(state.p_std, jnp.dot(params.weights, state.z), params)
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = self.update_std(state.std, norm_p_std, params)

        # Update M matrix
        M = self.update_M(state.M, p_std, state.z, params)

        return state.replace(mean=mean, std=std, p_std=p_std, M=M)

    def update_M(
        self, M: jax.Array, p_std: jax.Array, z: jax.Array, params: Params
    ) -> jax.Array:
        """Update the transformation matrix M."""
        p_std_outer = jnp.outer(p_std, p_std)
        z_outer_w = jnp.einsum("i,ijk->jk", params.weights, jax.vmap(jnp.outer)(z, z))
        I = jnp.eye(self.num_dims)
        return M @ (
            I + params.c_1 / 2 * (p_std_outer - I) + params.c_mu / 2 * (z_outer_w - I)
        )  # Eq. (M11) in Figure 3
