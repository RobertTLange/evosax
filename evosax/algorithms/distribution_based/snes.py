"""Separable Natural Evolution Strategy (Wierstra et al., 2014).

Reference: https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import struct

from ...types import Fitness, Population, Solution
from .base import Params, State, metrics_fn
from .xnes import xNES


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    opt_state: optax.OptState
    lrate_std: float


@struct.dataclass
class Params(Params):
    std_init: float
    weights: jax.Array
    lrate_std_init: float
    use_adaptation_sampling: bool
    rho: float  # Significance level adaptation sampling
    c_prime: float  # Adaptation sampling step size


class SNES(xNES):
    """Separable Natural Evolution Strategy (SNES)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1.0),
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize SNES."""
        super().__init__(
            population_size, solution, optimizer, metrics_fn, **fitness_kwargs
        )

    @property
    def _default_params(self) -> Params:
        params = super()._default_params

        # Override the learning rate for std with SNES-specific value
        lrate_std_init = (3 + jnp.log(self.num_dims)) / (5 * jnp.sqrt(self.num_dims))

        return Params(
            std_init=params.std_init,
            weights=params.weights,
            lrate_std_init=lrate_std_init,
            use_adaptation_sampling=params.use_adaptation_sampling,
            rho=params.rho,
            c_prime=params.c_prime,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
            lrate_std=params.lrate_std_init,
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
        population = state.mean + state.std * z
        return population, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z = (population - state.mean) / state.std
        z = z[fitness.argsort()]

        # Compute grad
        grad_mean = -state.std * jnp.dot(params.weights, z)
        updates, opt_state = self.optimizer.update(grad_mean, state.opt_state)

        # Update mean
        mean = optax.apply_updates(state.mean, updates)

        # Compute grad for std
        grad_std = jnp.dot(params.weights, z**2 - 1)

        # Update std
        std = state.std * jnp.exp(0.5 * state.lrate_std * grad_std)

        # Adaptation sampling for std learning rate
        lrate_std = self.adaptation_sampling(
            state.lrate_std,
            z,
            state,
            params,
        )
        lrate_std = jnp.where(
            params.use_adaptation_sampling, lrate_std, state.lrate_std
        )

        return state.replace(
            mean=mean, std=std, opt_state=opt_state, lrate_std=lrate_std
        )

    def adaptation_sampling(
        self,
        lrate_std: float,
        z_sorted: jax.Array,
        state: State,
        params: Params,
    ) -> float:
        """Adaptation sampling for std learning rate adaptation."""
        # Create hypothetical distribution with increased learning rate
        lrate_std_prime = 1.5 * lrate_std

        # Calculate what the std would be with the new learning rate
        grad_std = jnp.dot(params.weights, z_sorted**2 - 1)
        std_prime = state.std * jnp.exp(0.5 * lrate_std_prime * grad_std)

        # For the separable case, we need to handle per-dimension calculations
        # Calculate log ratios for each dimension, then sum across dimensions
        log_det_ratio = jnp.sum(jnp.log(state.std) - jnp.log(std_prime))

        # Calculate the exponent term for each sample and dimension
        std_ratio_squared = (state.std / std_prime) ** 2
        log_exp_terms = 0.5 * jnp.einsum("ij,j->i", z_sorted**2, std_ratio_squared - 1)

        # Combine for final log weights
        log_weights = log_det_ratio + log_exp_terms

        # Convert back to normal space for the Mann-Whitney test
        weights = jnp.exp(log_weights)

        # Perform weighted Mann-Whitney U test
        ranks = jnp.arange(1, self.population_size + 1)

        # Calculate weighted ranks
        weighted_ranks = weights * ranks
        sum_w = jnp.sum(weights)
        sum_wr = jnp.sum(weighted_ranks)

        # Compute the U statistic and its expected value/variance under H0
        U = sum_wr - sum_w * (sum_w + 1) / 2  # Corrected formula
        E_U = self.population_size * sum_w / 2
        Var_U = self.population_size * sum_w * (self.population_size + sum_w + 1) / 12

        # Compute z-score and p-value
        z_score = (U - E_U) / jnp.sqrt(Var_U + 1e-8)
        p_value = jax.scipy.stats.norm.cdf(z_score)

        # Update learning rate based on test result
        new_lrate_std = jnp.where(
            p_value < params.rho,
            jnp.minimum((1 + params.c_prime) * lrate_std, 1.0),
            (1 - params.c_prime) * lrate_std + params.c_prime * params.lrate_std_init,
        )

        return new_lrate_std
