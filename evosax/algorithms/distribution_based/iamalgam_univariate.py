"""(Iterative) AMaLGaM (Bosman et al., 2013) - Diagonal Covariance.

Reference: https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from ...types import Population, Solution
from .base import metrics_fn
from .iamalgam_full import Params, State, iAMaLGaM_Full


class iAMaLGaM_Univariate(iAMaLGaM_Full):
    """Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated Density-Estimation Evolutionary Algorithm (iAMaLGaM) with diagonal covariance matrix."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize iAMaLGaM."""
        super().__init__(
            population_size,
            solution,
            metrics_fn,
            **fitness_kwargs,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            mean_shift=jnp.zeros(self.num_dims),
            C=jnp.ones(self.num_dims),
            nis_counter=0,
            c_mult=params.c_mult_init,
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
        key_sample, key_ams = jax.random.split(key)

        z = jax.random.normal(
            key_sample, (self.population_size, self.num_dims)
        )  # ~ N(0, I)
        stds = state.std + jnp.sqrt(state.C)
        y = stds * z  # ~ N(0, C)
        population = state.mean + y  # ~ N(m, σ^2 C)

        population_ams = self.anticipated_mean_shift(
            key_ams,
            population,
            state.c_mult,
            state.mean_shift,
            params,
        )
        return population_ams, state

    def standard_deviation_ratio(
        self,
        improvement_mask: jax.Array,
        elites: jax.Array,
        C: jax.Array,
        mean: jax.Array,
    ) -> float:
        """SDR: relate distance of improvements to mean in search space."""
        # Compute average solutions that improved fitness
        solution_avg_imp = jnp.dot(improvement_mask, elites) / jnp.sum(improvement_mask)

        # Compute SDR
        conditioned_diff = (solution_avg_imp - mean) / C
        sdr = jnp.max(jnp.abs(conditioned_diff))
        return sdr

    def update_cov(
        self,
        elites: jax.Array,
        C: jax.Array,
        mean: jax.Array,
        params: Params,
    ) -> jax.Array:
        """Update covariance."""
        S_bar = elites - mean
        new_C = (1 - params.eta_std) * C + params.eta_std * jnp.sum(
            S_bar**2, axis=0
        ) / self.num_elites
        return new_C
