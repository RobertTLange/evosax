"""Matrix Adaptation Evolution Strategy (Bayer & Sendhoff, 2017).

[1] https://ieeexplore.ieee.org/document/7875115
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import State as BaseState, metrics_fn
from .cma_es import CMA_ES, Params as BaseParams


@struct.dataclass
class State(BaseState):
    mean: jax.Array
    std: jax.Array
    p_std: jax.Array
    M: jax.Array
    z: jax.Array


@struct.dataclass
class Params(BaseParams):
    pass


class MA_ES(CMA_ES):
    """Matrix Adaptation Evolution Strategy (MA-ES).

    Variant of CMA-ES that adapts a transformation matrix `M` directly instead of
    a covariance matrix. Samples are drawn as `mean + std * z @ M.T` with standard
    normal `z`; the step size is adapted with cumulative step length adaptation.
    Negative recombination weights are not used.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Bayer & Sendhoff (2017)](https://ieeexplore.ieee.org/document/7875115)
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        self.use_negative_weights = False

    def _init(  # type: ignore[override]
        self, key: jax.Array, params: Params
    ) -> State:
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

    def _ask(  # type: ignore[override]
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        # Sample new population
        z = jax.random.normal(key, (self.population_size, self.num_dims))
        population = state.mean + state.std * z @ state.M.T

        return population, state.replace(z=z)

    def _tell(  # type: ignore[override]
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update mean
        mean, _, _ = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # Cumulative Step length Adaptation (CSA)
        p_std = self.update_p_std(state.p_std, jnp.dot(fitness, state.z), params)
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = self.update_std(state.std, norm_p_std, params)

        # Update M matrix
        M = self.update_M(state.M, p_std, state.z, params)

        return state.replace(mean=mean, std=std, p_std=p_std, M=M)

    def update_M(
        self, M: jax.Array, p_std: jax.Array, z: jax.Array, params: Params
    ) -> jax.Array:
        """Update the transformation matrix M.

        Args:
            M: Current transformation matrix, shape `(num_dims, num_dims)`.
            p_std: Evolution path for the step-size adaptation, shape
                `(num_dims,)`.
            z: Standard normal samples drawn in the last `ask`, shape
                `(population_size, num_dims)`.
            params: Algorithm hyperparameters.

        Returns:
            The updated transformation matrix, shape `(num_dims, num_dims)`.
        """
        p_std_outer = jnp.outer(p_std, p_std)
        z_outer_w = jnp.einsum("i,ijk->jk", params.weights, jax.vmap(jnp.outer)(z, z))
        identity = jnp.eye(self.num_dims)
        return M @ (
            identity
            + params.c_1 / 2 * (p_std_outer - identity)
            + params.c_mu / 2 * (z_outer_w - identity)
        )  # Eq. (M11) in Figure 3
