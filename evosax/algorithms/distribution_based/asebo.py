"""Adaptive ES-Active Subspaces for Blackbox Optimization (Choromanski et al., 2019).

[1] https://arxiv.org/abs/1903.04268

Note that there are a couple of adaptations:
1. We always sample a fixed population size per generation
2. We keep a fixed archive of gradients to estimate the subspace
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    opt_state: optax.OptState
    grad_subspace: jax.Array
    alpha: float
    UUT: jax.Array
    UUT_ort: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    grad_decay: float


class ASEBO(DistributionBasedAlgorithm):
    """Adaptive ES-Active Subspaces for Blackbox Optimization (ASEBO)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,
        optimizer: optax.GradientTransformation = optax.adam(learning_rate=1e-3),
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

    @property
    def _default_params(self) -> Params:
        return Params(
            std_init=1.0,
            grad_decay=0.99,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
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
        X = state.grad_subspace
        X -= jnp.mean(X, axis=0)
        U, S, Vt = jnp.linalg.svd(X, full_matrices=False)

        def svd_flip(u, v):
            # columns of u, rows of v
            max_abs_cols = jnp.argmax(jnp.abs(u), axis=0)
            signs = jnp.sign(u[max_abs_cols, jnp.arange(u.shape[1])])
            u *= signs
            v *= signs[:, jnp.newaxis]
            return u, v

        U, Vt = svd_flip(U, Vt)
        U = Vt[: int(self.population_size / 2)]
        UUT = jnp.matmul(U.T, U)

        U_ort = Vt[int(self.population_size / 2) :]
        UUT_ort = jnp.matmul(U_ort.T, U_ort)

        subspace_ready = state.generation_counter > self.subspace_dims

        UUT = jax.lax.select(
            subspace_ready, UUT, jnp.zeros((self.num_dims, self.num_dims))
        )
        cov = (
            state.std * (state.alpha / self.num_dims) * jnp.eye(self.num_dims)
            + ((1 - state.alpha) / int(self.population_size / 2)) * UUT
        )
        chol = jnp.linalg.cholesky(cov)
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z_plus = z_plus @ chol.T
        z_plus /= jnp.linalg.norm(z_plus, axis=-1)[:, None]
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
            (population[: self.population_size // 2] - state.mean) / state.std,
        )

        alpha = jnp.linalg.norm(jnp.dot(grad, state.UUT_ort)) / jnp.linalg.norm(
            jnp.dot(grad, state.UUT)
        )
        subspace_ready = state.generation_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)

        # FIFO grad subspace (same as in guided_es.py)
        grad_subspace = jnp.roll(state.grad_subspace, shift=-1, axis=0)
        grad_subspace = grad_subspace.at[-1, :].set(grad)

        # Normalize gradients by norm / num_dims
        grad /= jnp.linalg.norm(grad) / self.num_dims + 1e-8

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean, opt_state=opt_state, grad_subspace=grad_subspace, alpha=alpha
        )
