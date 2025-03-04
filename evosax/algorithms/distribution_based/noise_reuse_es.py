"""Noise-Reuse Evolution Strategy (Li et al., 2023).

Reference: https://arxiv.org/abs/2304.12180
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ...core.fitness_shaping import identity_fitness_shaping_fn
from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    opt_state: optax.OptState
    pert: jax.Array  # Perturbations used in partial unroll multiple times
    inner_step_counter: int  # Keep track of unner unroll steps for noise reset


@struct.dataclass
class Params(Params):
    std_init: float
    T: int  # Total inner problem length
    K: int  # Truncation length for partial unrolls


class NoiseReuseES(DistributionBasedAlgorithm):
    """Noise-Reuse Evolution Strategy (NRES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize NRES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            T=100,
            K=10,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            pert=jnp.zeros((self.population_size, self.num_dims)),
            inner_step_counter=0,
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        # Antithetic sampling
        pert_pos = state.std * jax.random.normal(
            key, (self.population_size // 2, self.num_dims)
        )
        pert = jnp.concatenate([pert_pos, -pert_pos])

        # Sample each ask call but only use when trajectory is reset
        pert = jnp.where(state.inner_step_counter == 0, pert, state.pert)

        population = state.mean + pert
        return population, state.replace(pert=pert)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Compute grad
        grad = jnp.dot(fitness, state.pert) / state.std**2 / self.population_size

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Update inner step counter
        inner_step_counter = state.inner_step_counter + params.K

        # Resample perturbations in ask if done with inner problem
        inner_step_counter = jnp.where(
            inner_step_counter >= params.T, 0, inner_step_counter
        )

        return state.replace(
            mean=mean,
            opt_state=opt_state,
            inner_step_counter=inner_step_counter,
        )
