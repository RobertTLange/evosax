"""Limited Memory Matrix Adaptation Evolution Strategy (Loshchilov et al., 2017).

[1] https://arxiv.org/abs/1705.06693
Note: The original paper recommends a population size of 4 + 3 * jnp.log(num_dims).
Instabilities have been observed with larger population sizes.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from evosax.core.fitness_shaping import weights_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import metrics_fn
from .ma_es import MA_ES, Params, State


@struct.dataclass
class State(State):
    pass


@struct.dataclass
class Params(Params):
    c_d: jax.Array


class LM_MA_ES(MA_ES):
    """Limited Memory Matrix Adaptation Evolution Strategy (LM-MA-ES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        fitness_shaping_fn: Callable = weights_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize LM-MA-ES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        self.use_negative_weights = False

    @property
    def _default_params(self) -> Params:
        # Calculate m for LM-MA-ES
        self.m = int(4 + jnp.floor(3 * jnp.log(self.num_dims)))

        # Get parent class parameters
        parent_params = super()._default_params

        # Override or add LM-MA-ES specific parameters
        c_d = 1 / jnp.power(1.5, jnp.arange(self.m)) / self.num_dims
        c_c = self.population_size / jnp.power(4, jnp.arange(self.m)) / self.num_dims

        # Set c_1 and c_mu to 0 as they're not used in LM-MA-ES
        return Params(
            std_init=parent_params.std_init,
            std_min=parent_params.std_min,
            std_max=parent_params.std_max,
            weights=parent_params.weights,
            mu_eff=parent_params.mu_eff,
            c_mean=parent_params.c_mean,
            c_std=parent_params.c_std,
            d_std=parent_params.d_std,
            c_c=c_c,
            c_1=0.0,  # Not used in LM-MA-ES
            c_mu=0.0,  # Not used in LM-MA-ES
            chi_n=parent_params.chi_n,
            c_d=c_d,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            p_std=jnp.zeros(self.num_dims),
            M=jnp.zeros((self.m, self.num_dims)),
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

        d = z
        # L9-10
        for i in range(self.m):
            d = jnp.where(
                i < state.generation_counter,
                (1 - params.c_d[i]) * d
                + params.c_d[i] * jnp.dot(d, state.M[i])[:, None] * state.M[i],
                d,
            )

        population = state.mean + state.std * d
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Update mean
        # Use the parent method to update the mean
        mean, _, _ = self.update_mean(
            population, fitness, state.mean, state.std, params
        )

        # Cumulative Step length Adaptation (CSA) - reuse from parent class
        p_std = self.update_p_std(state.p_std, jnp.dot(fitness, state.z), params)
        norm_p_std = jnp.linalg.norm(p_std)

        # Update std
        std = self.update_std(state.std, norm_p_std, params)

        # Update M matrix - specific to LM-MA-ES
        M = self.update_M(state.M, state.z, params)

        return state.replace(mean=mean, std=std, p_std=p_std, M=M)

    def update_M(self, M: jax.Array, z: jax.Array, params: Params) -> jax.Array:
        """Update the low-memory transformation matrix M."""
        return (1 - params.c_c)[:, None] * M + jnp.sqrt(
            params.mu_eff * params.c_c * (2 - params.c_c)
        )[:, None] * jnp.dot(params.weights, z)
