"""Adaptive ES-Active Subspaces for Blackbox Optimization (Choromanski et al., 2019).

Reference: https://arxiv.org/abs/1903.04268

Note that there are a couple of adaptations:
1. We always sample a fixed population size per generation
2. We keep a fixed archive of gradients to estimate the subspace
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
    alpha: float
    UUT: jax.Array
    UUT_ort: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_decay: float
    std_limit: float
    opt_params: OptParams
    grad_decay: float


class ASEBO(DistributionBasedAlgorithm):
    """Adaptive ES-Active Subspaces for Blackbox Optimization (ASEBO)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        subspace_dims: int = 1,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize ASEBO."""
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
            std_limit=0.0,
            opt_params=opt_params,
            grad_decay=0.99,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))

        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            opt_state=self.optimizer.init(params.opt_params),
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
        # Reconstruct noise from last mean/std estimates
        noise = (population - state.mean) / state.std
        noise_1 = noise[: int(self.population_size / 2)]
        fit_1 = fitness[: int(self.population_size / 2)]
        fit_2 = fitness[int(self.population_size / 2) :]
        fit_diff_noise = jnp.dot(noise_1.T, fit_1 - fit_2)
        theta_grad = 1.0 / 2.0 * fit_diff_noise

        alpha = jnp.linalg.norm(jnp.dot(theta_grad, state.UUT_ort)) / jnp.linalg.norm(
            jnp.dot(theta_grad, state.UUT)
        )
        subspace_ready = state.generation_counter > self.subspace_dims
        alpha = jax.lax.select(subspace_ready, alpha, 1.0)

        # Add grad FIFO-style to subspace archive (only if provided else FD)
        grad_subspace = jnp.zeros((self.subspace_dims, self.num_dims))
        grad_subspace = grad_subspace.at[:-1, :].set(state.grad_subspace[1:, :])
        grad_subspace = grad_subspace.at[-1, :].set(theta_grad)
        state = state.replace(grad_subspace=grad_subspace)

        # Normalize gradients by norm / num_dims
        theta_grad /= jnp.linalg.norm(theta_grad) / self.num_dims + 1e-8

        # Grad update using optimizer instance - decay lrate if desired
        mean, opt_state = self.optimizer.step(
            state.mean, theta_grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update lrate and standard deviation based on min and decay
        std = jnp.clip(state.std * params.std_decay, min=params.std_limit)
        return state.replace(mean=mean, std=std, opt_state=opt_state, alpha=alpha)
