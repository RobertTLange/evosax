"""Guided Evolution Strategy (Maheswaranathan et al., 2018).

[1] https://arxiv.org/abs/1806.10230
[2] https://github.com/brain-research/guided-evolutionary-strategies
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
    grad: jax.Array
    z: jax.Array


@struct.dataclass
class Params(BaseParams):
    alpha: float
    beta: float


class GuidedES(DistributionBasedAlgorithm):
    """Guided Evolution Strategy (GuidedES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,  # k param in example notebook
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize GuidedES."""
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
        return Params(alpha=0.5, beta=1.0)

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            grad=jnp.full((self.num_dims,), jnp.nan),
            grad_subspace=jnp.zeros((self.num_dims, self.subspace_dims)),
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
        """Ask evolution strategy for new candidate solutions to evaluate.

        Unlike traditional evolution strategies, Guided ES requires a gradient to
        generate new solutions. Before calling this method, update the gradient in the
        state using state.replace(grad=grad). If no gradient is provided, a zero
        gradient will be used.
        """
        a = state.std * jnp.sqrt(params.alpha / self.num_dims)
        c = state.std * jnp.sqrt((1.0 - params.alpha) / self.subspace_dims)

        # FIFO grad subspace
        grad = jnp.where(jnp.isnan(state.grad), 0.0, state.grad)
        grad_subspace = jnp.roll(state.grad_subspace, shift=-1, axis=-1)
        grad_subspace = grad_subspace.at[:, -1].set(grad)

        # Sample noise guided by grad subspace
        key_full, key_sub = jax.random.split(key, 2)
        eps_full = jax.random.normal(
            key_full, shape=(self.num_dims, self.population_size // 2)
        )
        eps_subspace = jax.random.normal(
            key_sub, shape=(self.subspace_dims, self.population_size // 2)
        )
        Q, _ = jnp.linalg.qr(grad_subspace)

        # Antithetic sampling
        z_plus = a * eps_full + c * jnp.dot(Q, eps_subspace)
        z_plus = jnp.transpose(z_plus)
        z = jnp.concatenate([z_plus, -z_plus])
        x = state.mean + z
        return x, state.replace(
            grad=jnp.full((self.num_dims,), jnp.nan), grad_subspace=grad_subspace, z=z
        )

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Compute gradient
        z_plus = state.z[: self.population_size // 2]
        fitness_plus = fitness[: self.population_size // 2]
        fitness_minus = fitness[self.population_size // 2 :]

        # Compute grad
        delta = fitness_plus - fitness_minus
        grad = params.beta * jnp.dot(delta, z_plus) / (2 * state.std**2)

        # Update mean
        updates, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            opt_state=opt_state,
        )
