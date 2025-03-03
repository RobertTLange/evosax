"""Guided Evolution Strategy (Maheswaranathan et al., 2018).

Reference: https://arxiv.org/abs/1806.10230
Inspired by: https://github.com/brain-research/guided-evolutionary-strategies
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...core import GradientOptimizer, OptParams, OptState
from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    opt_state: OptState
    grad_subspace: jax.Array
    grad: jax.Array
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_decay: float
    std_limit: float
    opt_params: OptParams
    alpha: float
    beta: float


class GuidedES(DistributionBasedAlgorithm):
    """Guided Evolution Strategy (GuidedES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,  # k param in example notebook
        opt_name: str = "sgd",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize GuidedES."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        assert subspace_dims <= self.num_dims, (
            "Subspace dims must be smaller than optimization dims."
        )
        self.subspace_dims = subspace_dims

        # Optimizer
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit

    @property
    def _default_params(self) -> Params:
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return Params(
            std_init=1.0,
            std_decay=1.0,
            std_limit=0.01,
            opt_params=opt_params,
            alpha=0.5,
            beta=1.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            opt_state=self.optimizer.init(params.opt_params),
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

        delta = fitness_plus - fitness_minus
        grad = params.beta * jnp.dot(z_plus.T, delta) / (2 * state.std**2)

        # Grad update using optimizer
        mean, opt_state = self.optimizer.step(
            state.mean, grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update state
        std = jnp.clip(state.std * params.std_decay, min=params.std_limit)
        return state.replace(mean=mean, std=std, opt_state=opt_state)
