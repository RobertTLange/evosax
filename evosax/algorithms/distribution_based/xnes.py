"""Exponential Natural Evolution Strategy (Wierstra et al., 2014).

Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
Inspired by: https://github.com/chanshing/xnes
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
    B: jax.Array
    lr_std: float
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    lr_std_init: float
    lr_B: float
    use_adaptation_sampling: bool
    rho: float  # Significance level adaptation sampling
    c_prime: float  # Adaptation sampling step size


class xNES(DistributionBasedAlgorithm):
    """Exponential Natural Evolution Strategy (xNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize xNES."""
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        # Optimizer
        self.optimizer = optimizer

    @property
    def _default_params(self) -> Params:
        weights = get_weights(self.population_size)

        lr_std_init = (9 + 3 * jnp.log(self.num_dims)) / (
            5 * jnp.sqrt(self.num_dims) * self.num_dims
        )
        rho = 0.5 - 1 / (3 * (self.num_dims + 1))
        return Params(
            std_init=1.0,
            weights=weights,
            lr_std_init=lr_std_init,
            lr_B=lr_std_init,
            use_adaptation_sampling=False,
            rho=rho,
            c_prime=0.1,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            B=params.std_init * jnp.eye(self.num_dims),
            lr_std=params.lr_std_init,
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
        population = state.mean + state.std * z @ state.B
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z = state.z[fitness.argsort()]

        # Compute grad for mean
        grad_mean = -state.std * state.B @ jnp.dot(params.weights, z)

        # Update mean
        updates, opt_state = self.optimizer.update(grad_mean, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        # Compute grad for std
        grad_M = jnp.einsum(
            "i,ijk->jk",
            params.weights,
            jax.vmap(jnp.outer)(z, z) - jnp.eye(self.num_dims),
        )
        grad_std = jnp.trace(grad_M) / self.num_dims

        # Update std
        std = state.std * jnp.exp(0.5 * state.lr_std * grad_std)

        # Update B
        grad_B = grad_M - grad_std * jnp.eye(self.num_dims)
        B = state.B * jnp.exp(0.5 * params.lr_B * grad_B)

        # Adaptation sampling for std learning rate
        lr_std = self.adaptation_sampling(
            state.lr_std,
            z,
            state,
            params,
        )
        lr_std = jnp.where(params.use_adaptation_sampling, lr_std, state.lr_std)

        return state.replace(
            mean=mean, std=std, opt_state=opt_state, B=B, lr_std=lr_std
        )

    def adaptation_sampling(
        self,
        lr_std: float,
        z_sorted: jax.Array,
        state: State,
        params: Params,
    ) -> float:
        """Adaptation sampling for std learning rate adaptation."""
        # Create hypothetical distribution with increased learning rate
        lr_std_prime = 1.5 * lr_std

        # Calculate what the distribution would be with the new learning rate
        # Instead of computing the full matrix grad_M and then taking the trace
        # Directly compute the trace which is more efficient:
        weighted_squares_sum = jnp.einsum("i,ij->j", params.weights, z_sorted**2)
        grad_std = (
            jnp.sum(weighted_squares_sum) - self.num_dims * jnp.sum(params.weights)
        ) / self.num_dims

        # The hypothetical new std
        std_prime = state.std * jnp.exp(0.5 * lr_std_prime * grad_std)

        # Calculate importance weights (ratio of probability densities)
        # Original distribution: N(0, I) since z are already standardized samples
        # New distribution: we're changing the scale, not the shape
        log_ratio = self.num_dims * jnp.log(state.std / std_prime)
        log_exp_term = (
            0.5 * ((state.std / std_prime) ** 2 - 1) * jnp.sum(z_sorted**2, axis=1)
        )

        # Use logaddexp to stabilize addition in log space when needed
        log_weights = log_ratio + log_exp_term

        # Convert back to normal space for the rest of the computation
        weights = jnp.exp(log_weights)

        # Perform weighted Mann-Whitney U test
        ranks = jnp.arange(1, self.population_size + 1)

        # Calculate weighted ranks
        weighted_ranks = weights * ranks
        sum_w = jnp.sum(weights)
        sum_wr = jnp.sum(weighted_ranks)

        # Compute the U statistic and its expected value/variance under H0
        U = sum_wr - sum_w * (sum_w + 1) / 2
        E_U = self.population_size * sum_w / 2
        Var_U = self.population_size * sum_w * (self.population_size + sum_w + 1) / 12

        # Compute z-score and p-value
        z_score = (U - E_U) / jnp.sqrt(Var_U + 1e-8)
        p_value = jax.scipy.stats.norm.cdf(z_score)

        # Update learning rate based on test result
        # If p_value < rho (significant improvement), increase learning rate
        # Otherwise, move it closer to initial value
        new_lr_std = jnp.where(
            p_value < params.rho,
            jnp.minimum((1 + params.c_prime) * lr_std, 1.0),  # Cap at 1.0
            (1 - params.c_prime) * lr_std + params.c_prime * params.lr_std_init,
        )

        return new_lr_std


def get_weights(population_size: int):
    """Get weights for fitness shaping."""
    weights = jnp.clip(
        jnp.log(population_size / 2 + 1) - jnp.log(jnp.arange(1, population_size + 1)),
        min=0.0,
    )
    weights = weights / jnp.sum(weights)
    return weights - 1 / population_size
