"""(Iterative) AMaLGaM (Bosman et al., 2013) - Diagonal Covariance.

[1] https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Population, Solution

from .base import metrics_fn
from .iamalgam_full import Params, State, iAMaLGaM_Full


class iAMaLGaM_Univariate(iAMaLGaM_Full):
    """iAMaLGaM with diagonal covariance matrix.

    Incremental Adapted Maximum-Likelihood Gaussian Model - Iterated
    Density-Estimation Evolutionary Algorithm. Works like `iAMaLGaM_Full` but
    stores only the diagonal of the covariance matrix: the population is sampled
    independently per dimension and `C` is a vector of per-dimension variances.

    Args:
        population_size: Number of candidate solutions sampled and evaluated per
            generation.
        solution: Example solution PyTree. Only its structure and leaf shapes are
            used: they set the number of search dimensions and how flat search
            vectors are unravelled into the PyTrees returned by `ask`.
        std_schedule: Function mapping the generation counter to `std`, e.g. an
            Optax schedule. `std` is added to the per-dimension standard
            deviations `sqrt(C)` when sampling.
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
        super().__init__(
            population_size, solution, std_schedule, fitness_shaping_fn, metrics_fn
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=self.std_schedule(0),
            mean_shift=jnp.zeros(self.num_dims),
            C=jnp.ones(self.num_dims),
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
    ) -> jax.Array:
        """Compute the standard deviation ratio (SDR).

        Averages the elites that improved on the best fitness so far, divides
        the offset of that average from the mean elementwise by `C` and returns
        the largest absolute component.

        Args:
            improvement_mask: Boolean vector of shape `(num_elites,)` marking the
                elites that improved on the best fitness so far.
            elites: Elite solutions of shape `(num_elites, num_dims)`.
            C: Per-dimension variances of shape `(num_dims,)`.
            mean: Mean of the search distribution, shape `(num_dims,)`.

        Returns:
            The scalar standard deviation ratio.
        """
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
        """Update the per-dimension variances.

        Blends `C` with the per-dimension mean squared deviation of the elites
        from `mean` using rate `params.eta_std`.

        Args:
            elites: Elite solutions of shape `(num_elites, num_dims)`.
            C: Current per-dimension variances of shape `(num_dims,)`.
            mean: Mean the elites are centered on, shape `(num_dims,)`.
            params: Algorithm hyperparameters.

        Returns:
            The updated per-dimension variances.
        """
        S_bar = elites - mean
        new_C = (1 - params.eta_std) * C + params.eta_std * jnp.sum(
            S_bar**2, axis=0
        ) / self.num_elites
        return new_C
