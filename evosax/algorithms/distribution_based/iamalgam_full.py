"""Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated Density-Estimation
Evolutionary Algorithm (Bosman et al., 2013).

Full covariance version.

[1] https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from ..base import update_best_solution_and_fitness
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
    C: jax.Array
    mean_shift: jax.Array
    nis_counter: int | jax.Array
    c_mult: float | jax.Array
    best_solution_shaped: Solution
    best_fitness_shaped: float


@struct.dataclass
class Params(BaseParams):
    eta_std: jax.Array
    eta_shift: jax.Array
    eta_avs_inc: float
    eta_avs_dec: float
    nis_max_gens: int
    delta_ams: float
    theta_sdr: float
    c_mult_init: float


class iAMaLGaM_Full(DistributionBasedAlgorithm):
    """iAMaLGaM with full covariance matrix.

    Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated
    Density-Estimation Evolutionary Algorithm. Samples the population from a
    multivariate Gaussian, then shifts a random subset of it along the running
    mean shift (anticipated mean shift). `tell` sets the mean to the average of
    the best half of the population and incrementally updates the mean shift and
    the full covariance matrix from those elites. A distribution multiplier
    `c_mult` is adapted from whether the elites improve on the best fitness so
    far and from the standard deviation ratio; it scales the anticipated mean
    shift.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        std_schedule: Function mapping the generation counter to `std`, e.g. an
            Optax schedule. `std**2` is added to the diagonal of the covariance
            matrix `C` when sampling.
        fitness_shaping_fn: Function
            `(population, fitness, state, params) -> shaped_fitness` applied to
            the raw fitness in `tell` before the algorithm update. See
            `evosax.core.fitness_shaping`.
        metrics_fn: Function `(key, population, fitness, state, params) -> metrics`
            evaluated in `tell`; its output is returned alongside the new state.

    References:
        - [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf)
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        std_schedule: Callable = optax.constant_schedule(1.0),
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

        self.elite_ratio = 0.5
        alpha_ams = (
            0.5
            * self.elite_ratio
            * self.population_size
            / (self.population_size - self.num_elites)
        )
        self.ams_population_size = int(alpha_ams * (self.population_size - 1))

        # std schedule
        self.std_schedule = std_schedule

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
            std=self.std_schedule(0),
            mean_shift=jnp.zeros(self.num_dims),
            C=jnp.eye(self.num_dims),
            nis_counter=0,
            c_mult=params.c_mult_init,
            best_solution_shaped=jnp.full((self.num_dims,), jnp.nan),
            best_fitness_shaped=jnp.inf,
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
        improvement_mask = fitness_elites < state.best_fitness_shaped
        any_improvement = jnp.any(improvement_mask)

        # Standard deviation ratio
        sdr = self.standard_deviation_ratio(
            improvement_mask, elites, state.C, state.mean
        )

        # Adaptive variance scaling
        c_mult, nis_counter = self.adaptive_variance_scaling(
            any_improvement,
            sdr,
            cast(jax.Array, state.c_mult),
            cast(jax.Array, state.nis_counter),
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

        # Update best solution and fitness shaped
        best_solution_shaped, best_fitness_shaped = update_best_solution_and_fitness(
            population, fitness, state.best_solution_shaped, state.best_fitness_shaped
        )

        return state.replace(
            mean=mean,
            std=self.std_schedule(state.generation_counter),
            C=C,
            mean_shift=mean_shift,
            nis_counter=nis_counter,
            c_mult=c_mult,
            best_solution_shaped=best_solution_shaped,
            best_fitness_shaped=best_fitness_shaped,
        )

    def anticipated_mean_shift(
        self,
        key: jax.Array,
        population: jax.Array,
        c_mult: float | jax.Array,
        mean_shift: jax.Array,
        params: Params,
    ) -> jax.Array:
        """Apply the anticipated mean shift (AMS) to part of the population.

        Adds `c_mult * params.delta_ams * mean_shift` to `ams_population_size`
        members chosen at random without replacement.

        Args:
            key: PRNG key used to choose the shifted members.
            population: Flat population of shape `(population_size, num_dims)`.
            c_mult: Current distribution multiplier.
            mean_shift: Running estimate of the mean shift, shape `(num_dims,)`.
            params: Algorithm hyperparameters.

        Returns:
            The population with the chosen members shifted.
        """
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
    ) -> jax.Array:
        """Compute the standard deviation ratio (SDR).

        Averages the elites that improved on the best fitness so far, expresses
        the offset of that average from the mean in units of the Cholesky factor
        of `C` and returns the largest absolute component.

        Args:
            improvement_mask: Boolean vector of shape `(num_elites,)` marking the
                elites that improved on the best fitness so far.
            elites: Elite solutions of shape `(num_elites, num_dims)`.
            C: Covariance matrix of shape `(num_dims, num_dims)`.
            mean: Mean of the search distribution, shape `(num_dims,)`.

        Returns:
            The scalar standard deviation ratio.
        """
        # Compute average solutions that improved fitness
        solution_avg_imp = jnp.dot(improvement_mask, elites) / jnp.sum(improvement_mask)

        # Compute SDR
        L = jax.scipy.linalg.cholesky(C)
        conditioned_diff = jnp.linalg.inv(L) @ (solution_avg_imp - mean)
        sdr = jnp.max(jnp.abs(conditioned_diff))
        return sdr

    def adaptive_variance_scaling(
        self,
        any_improvement: jax.Array,
        sdr: jax.Array,
        c_mult: jax.Array,
        nis_counter: jax.Array,
        params: Params,
    ) -> tuple[jax.Array, jax.Array]:
        """Adapt the distribution multiplier (adaptive variance scaling, AVS).

        With an improvement, the no-improvement counter is reset, a multiplier
        below 1 is reset to 1 and the multiplier is scaled by
        `params.eta_avs_inc` if `sdr > params.theta_sdr`. Without an improvement,
        the counter is incremented while the multiplier is at most 1, and the
        multiplier is scaled by `params.eta_avs_dec` if it exceeds 1 or the
        counter has reached `params.nis_max_gens`.

        Args:
            any_improvement: Whether any elite improved on the best fitness so
                far.
            sdr: Standard deviation ratio from `standard_deviation_ratio`.
            c_mult: Current distribution multiplier.
            nis_counter: Current no-improvement counter.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(c_mult, nis_counter)` with the updated multiplier and
            counter.
        """
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
        """Update the mean and the running mean shift.

        The new mean is the average of the elites. The mean shift is an
        exponential moving average of the mean displacement with rate
        `params.eta_shift`.

        Args:
            elites: Elite solutions of shape `(num_elites, num_dims)`.
            mean: Current mean, shape `(num_dims,)`.
            mean_shift: Current mean shift, shape `(num_dims,)`.
            params: Algorithm hyperparameters.

        Returns:
            A tuple `(mean, mean_shift)` with the updated values.
        """
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
        """Update the full covariance matrix.

        Blends `C` with the covariance of the elites around `mean` using rate
        `params.eta_std`.

        Args:
            elites: Elite solutions of shape `(num_elites, num_dims)`.
            C: Current covariance matrix of shape `(num_dims, num_dims)`.
            mean: Mean the elites are centered on, shape `(num_dims,)`.
            params: Algorithm hyperparameters.

        Returns:
            The updated covariance matrix.
        """
        S_bar = elites - mean
        new_C = (1 - params.eta_std) * C + params.eta_std * (
            S_bar.T @ S_bar
        ) / self.num_elites
        return new_C
