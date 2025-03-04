"""Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated Density-Estimation Evolutionary Algorithm (Bosman et al., 2013).

Full covariance version.

Reference: https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: float
    C: jax.Array
    mean_shift: jax.Array
    nis_counter: int
    c_mult: float


@struct.dataclass
class Params(Params):
    std_init: float
    eta_std: float
    eta_shift: float
    eta_avs_inc: float
    eta_avs_dec: float
    nis_max_gens: int
    delta_ams: float
    theta_sdr: float
    c_mult_init: float


class iAMaLGaM_Full(DistributionBasedAlgorithm):
    """Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated Density-Estimation Evolutionary Algorithm (iAMaLGaM) with full covariance matrix."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize iAMaLGaM."""
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        self.elite_ratio = 0.5
        alpha_ams = (
            0.5
            * self.elite_ratio
            * self.population_size
            / (self.population_size - self.num_elites)
        )
        self.ams_population_size = int(alpha_ams * (self.population_size - 1))

    @property
    def _default_params(self) -> Params:
        # Table 1
        a_0_std, a_1_std, a_2_std = -1.1, 1.2, 1.6
        a_0_shift, a_1_shift, a_2_shift = -1.2, 0.31, 0.5

        eta_std = 1 - jnp.exp(
            a_0_std * self.num_elites**a_1_std / self.num_dims**a_2_std
        )
        eta_shift = 1 - jnp.exp(
            a_0_shift * self.num_elites**a_1_shift / self.num_dims**a_2_shift
        )

        return Params(
            std_init=1.0,
            eta_std=eta_std,
            eta_shift=eta_shift,
            eta_avs_inc=1 / 0.9,
            eta_avs_dec=0.9,
            nis_max_gens=50,
            delta_ams=2.0,
            theta_sdr=1.0,
            c_mult_init=1.0,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init,
            mean_shift=jnp.zeros(self.num_dims),
            C=jnp.eye(self.num_dims),
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

        S = state.C + state.std**2 * jnp.eye(self.num_dims)
        population = jax.random.multivariate_normal(
            key_sample, state.mean, S, (self.population_size,)
        )  # ~ N(m, S) - shape: (population_size, num_dims)

        population_ams = self.anticipated_mean_shift(
            key_ams,
            population,
            state.c_mult,
            state.mean_shift,
            params,
        )
        return population_ams, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        # Sort
        idx = jnp.argsort(fitness)[: self.num_elites]
        elites = population[idx]
        fitness_elites = fitness[idx]

        # Check for fitness improvement
        improvement_mask = fitness_elites < state.best_fitness
        any_improvement = jnp.any(improvement_mask)

        # Standard deviation ratio
        sdr = self.standard_deviation_ratio(
            improvement_mask, elites, state.C, state.mean
        )

        # Adaptive variance scaling
        c_mult, nis_counter = self.adaptive_variance_scaling(
            any_improvement,
            sdr,
            state.c_mult,
            state.nis_counter,
            params,
        )

        # Update mean
        mean, mean_shift = self.update_mean(
            elites,
            state.mean,
            state.mean_shift,
            params,
        )

        # Update covariance - difference full vs. indep
        C = self.update_cov(elites, state.C, mean, params)

        return state.replace(
            mean=mean,
            C=C,
            mean_shift=mean_shift,
            nis_counter=nis_counter,
            c_mult=c_mult,
        )

    def anticipated_mean_shift(
        self,
        key: jax.Array,
        population: jax.Array,
        c_mult: float,
        mean_shift: jax.Array,
        params: Params,
    ) -> jax.Array:
        """AMS: Move some individuals in the direction of anticipated improvement."""
        idx = jax.random.choice(
            key, self.population_size, (self.ams_population_size,), replace=False
        )
        population_ams = population.at[idx].add(c_mult * params.delta_ams * mean_shift)
        return population_ams

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
        L = jax.scipy.linalg.cholesky(C)
        conditioned_diff = jnp.linalg.inv(L) @ (solution_avg_imp - mean)
        sdr = jnp.max(jnp.abs(conditioned_diff))
        return sdr

    def adaptive_variance_scaling(
        self,
        any_improvement: bool,
        sdr: float,
        c_mult: float,
        nis_counter: int,
        params: Params,
    ) -> tuple[float, int]:
        """AVS - adaptively rescale covariance depending on SDR."""
        # Case 1: If improvement in best fitness -> SDR increase c_mult! L14-19
        new_nis_counter = jnp.where(any_improvement, 0, nis_counter)
        reset_cond = jnp.logical_and(any_improvement, c_mult < 1)
        c_mult = jnp.where(reset_cond, 1.0, c_mult)

        std_cond = jnp.logical_and(any_improvement, sdr > params.theta_sdr)
        c_mult_inc = jnp.where(std_cond, params.eta_avs_inc * c_mult, c_mult)

        # Case 2: If  no improvement in best fitness -> Decrease c_mult! L21-24
        nis_dec_cond = jnp.logical_and(~any_improvement, c_mult <= 1)
        new_nis_counter = jnp.where(nis_dec_cond, nis_counter + 1, new_nis_counter)

        dec_cond = jnp.logical_and(
            ~any_improvement,
            jnp.logical_or(c_mult > 1, new_nis_counter >= params.nis_max_gens),
        )
        c_mult_dec = jnp.where(dec_cond, params.eta_avs_dec * c_mult, c_mult)

        c_dec_cond = jnp.logical_and(
            ~any_improvement, new_nis_counter < params.nis_max_gens
        )
        c_dec_reset_cond = jnp.logical_and(c_dec_cond, c_mult_dec < 1)
        c_mult_dec = jnp.where(c_dec_reset_cond, 1.0, c_mult_dec)

        # Select new multiplier based on case at hand
        new_c_mult = jnp.where(any_improvement, c_mult_inc, c_mult_dec)
        return new_c_mult, new_nis_counter

    def update_mean(
        self,
        elites: jax.Array,
        mean: jax.Array,
        mean_shift: jax.Array,
        params: Params,
    ) -> tuple[jax.Array, jax.Array]:
        """Update mean and mean shift."""
        new_mean = jnp.mean(elites, axis=0)
        new_mean_shift = (1 - params.eta_shift) * mean_shift + params.eta_shift * (
            new_mean - mean
        )
        return new_mean, new_mean_shift

    def update_cov(
        self,
        elites: jax.Array,
        C: jax.Array,
        mean: jax.Array,
        params: Params,
    ) -> jax.Array:
        """Update covariance."""
        S_bar = elites - mean
        new_C = (1 - params.eta_std) * C + params.eta_std * (
            S_bar.T @ S_bar
        ) / self.num_elites
        return new_C
