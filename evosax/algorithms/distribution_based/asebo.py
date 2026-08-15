"""Adaptive ES-Active Subspaces for Blackbox Optimization (Choromanski et al., 2019).

[1] https://arxiv.org/abs/1903.04268

Note that there are a couple of adaptations:
1. We always sample a fixed population size per generation
2. We keep a fixed archive of gradients to estimate the subspace

Alpha is clamped to [0, 1] for numerical stability. On high-dimensional noisy
problems this often saturates at 1 until the gradient archive is informative.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
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
    std: float
    opt_state: optax.OptState
    grad_subspace: jax.Array
    alpha: float
    UUT: jax.Array
    UUT_ort: jax.Array


@struct.dataclass
class Params(BaseParams):
    grad_decay: float


class ASEBO(DistributionBasedAlgorithm):
    """Adaptive ES-Active Subspaces for Blackbox Optimization (ASEBO)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize ASEBO."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        assert subspace_dims <= self.num_dims, (
            "Subspace dims must be smaller than optimization dims."
        )
        self.subspace_dims = subspace_dims

        # Optimizer
        self.optimizer = optimizer

        # std schedule
        self.std_schedule = std_schedule

    @property
    def _default_params(self) -> Params:
        return Params(grad_decay=0.99)

    def _init(self, key: jax.Array, params: Params) -> State:
        grad_subspace = jnp.zeros((self.subspace_dims + 1, self.num_dims))

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            grad_subspace=grad_subspace,
            alpha=1.0,
            UUT=jnp.zeros((self.num_dims, self.num_dims)),
            UUT_ort=jnp.zeros((self.num_dims, self.num_dims)),
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
        # Antithetic sampling of noise
        centered_gradients = state.grad_subspace - jnp.mean(state.grad_subspace, axis=0)
        _, singular_values, right_singular_vectors = jnp.linalg.svd(
            centered_gradients, full_matrices=False
        )
        subspace_basis = right_singular_vectors[: self.subspace_dims]
        rank_tolerance = (
            jnp.finfo(singular_values.dtype).eps
            * max(centered_gradients.shape)
            * singular_values[0]
        )
        informative_directions = (
            singular_values[: self.subspace_dims] > rank_tolerance
        ).astype(subspace_basis.dtype)
        UUT = (subspace_basis.T * informative_directions) @ subspace_basis
        effective_subspace_dims = jnp.maximum(jnp.sum(informative_directions), 1.0)
        # Orthogonal projector I - UU^T (avoids empty U_ort when subspace_dims < pop/2)
        UUT_ort = jnp.eye(self.num_dims) - UUT

        subspace_ready = state.generation_counter > self.subspace_dims

        UUT = jax.lax.select(
            subspace_ready, UUT, jnp.zeros((self.num_dims, self.num_dims))
        )
        UUT_ort = jax.lax.select(
            subspace_ready, UUT_ort, jnp.zeros((self.num_dims, self.num_dims))
        )
        cov = (state.alpha / self.num_dims) * jnp.eye(self.num_dims) + (
            (1 - state.alpha) / effective_subspace_dims
        ) * UUT
        # Jitter keeps cov SPD when alpha is small / subspace is low-rank
        cov = cov + 1e-6 * jnp.eye(self.num_dims)
        chol = jnp.linalg.cholesky(cov)
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z_plus = state.std * (z_plus @ chol.T)
        z = jnp.concatenate([z_plus, -z_plus])
        population = state.mean + z
        return population, state.replace(UUT=UUT, UUT_ort=UUT_ort)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Compute grad
        fitness_plus = fitness[: self.population_size // 2]
        fitness_minus = fitness[self.population_size // 2 :]
        grad = 0.5 * jnp.dot(
            fitness_plus - fitness_minus,
            (population[: self.population_size // 2] - state.mean) / state.std**2,
        )

        # Clamp alpha in [0, 1]; epsilon avoids 0/0 when projectors are empty/singular
        alpha = jnp.linalg.norm(jnp.dot(grad, state.UUT_ort)) / (
            jnp.linalg.norm(jnp.dot(grad, state.UUT)) + 1e-8
        )
        alpha = jnp.clip(alpha, 0.0, 1.0)
        subspace_ready = state.generation_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)

        # FIFO grad subspace (same as in guided_es.py)
        grad_subspace = jnp.roll(state.grad_subspace, shift=-1, axis=0)
        grad_subspace = grad_subspace.at[-1, :].set(grad)

        # Normalize gradients by norm / num_dims
        grad /= jnp.linalg.norm(grad) / self.num_dims + 1e-8

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state, state.mean)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
            grad_subspace=grad_subspace,
            alpha=alpha,
        )
