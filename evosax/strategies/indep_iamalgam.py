import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from .full_iamalgam import (
    anticipated_mean_shift,
    adaptive_variance_scaling,
    update_mean_amalgam,
)


class Indep_iAMaLGaM(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.35):
        """(Iterative) AMaLGaM (Bosman et al., 2013) - Diagonal Covariance
        Reference: https://tinyurl.com/y9fcccx2
        """
        super().__init__(num_dims, popsize)
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        alpha_ams = (
            0.5
            * self.elite_ratio
            * self.popsize
            / (self.popsize - self.elite_popsize)
        )
        self.ams_popsize = int(alpha_ams * (self.popsize - 1))
        self.strategy_name = "Indep_iAMaLGaM"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        a_0_sigma, a_1_sigma, a_2_sigma = -1.1, 1.2, 1.6
        a_0_shift, a_1_shift, a_2_shift = -1.2, 0.31, 0.5
        eta_sigma = 1 - jnp.exp(
            a_0_sigma
            * self.elite_popsize ** a_1_sigma
            / (self.num_dims ** a_2_sigma)
        )
        eta_shift = 1 - jnp.exp(
            a_0_shift
            * self.elite_popsize ** a_1_shift
            / (self.num_dims ** a_2_shift)
        )

        params = {
            "eta_sigma": eta_sigma,
            "eta_shift": eta_shift,
            "eta_avs_inc": 1.0 / 0.9,
            "eta_avs_dec": 0.9,
            "nis_max_gens": 50,
            "delta_ams": 2.0,
            "theta_sdr": 1.0,
            "c_mult_init": 1.0,
            "sigma_init": 0.0,
            "sigma_decay": 0.999,
            "sigma_limit": 0.0,
            "init_min": 0.0,
            "init_max": 0.0,
        }
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        # Initialize evolution paths & covariance matrix
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "mean": initialization,
            "mean_shift": jnp.zeros(self.num_dims),
            "C": jnp.ones(self.num_dims),
            "sigma": params["sigma_init"],
            "nis_counter": 0,
            "c_mult": params["c_mult_init"],
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        rng_sample, rng_ams = jax.random.split(rng)
        x = sample(
            rng_sample, state["mean"], state["C"], state["sigma"], self.popsize
        )
        x_ams = anticipated_mean_shift(
            rng_ams,
            x,
            self.ams_popsize,
            params["delta_ams"],
            state["c_mult"],
            state["mean_shift"],
        )
        return x_ams, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite
        idx = jnp.argsort(fitness)[0 : self.elite_popsize]
        fitness_elite = fitness[idx]
        members_elite = x[idx]

        # If there has been a fitness improvement -> Run AVS based on SDR
        improvements = fitness_elite < state["best_fitness"]
        any_improvement = jnp.sum(improvements) > 0
        sdr = standard_deviation_ratio(
            improvements, members_elite, state["mean"], state["C"]
        )
        state["c_mult"], state["nis_counter"] = adaptive_variance_scaling(
            any_improvement,
            sdr,
            state["c_mult"],
            state["nis_counter"],
            params["theta_sdr"],
            params["eta_avs_inc"],
            params["eta_avs_dec"],
            params["nis_max_gens"],
        )

        # Update mean and covariance estimates - difference full vs. indep
        state["mean"], state["mean_shift"] = update_mean_amalgam(
            members_elite,
            state["mean"],
            state["mean_shift"],
            params["eta_shift"],
        )
        state["C"] = update_cov_amalgam(
            members_elite, state["C"], state["mean"], params["eta_sigma"]
        )

        # Decay isotropic part of Gaussian search distribution
        state["sigma"] *= params["sigma_decay"]
        state["sigma"] = jnp.maximum(state["sigma"], params["sigma_limit"])
        return state


def sample(
    rng: chex.PRNGKey,
    mean: chex.Array,
    C: chex.Array,
    sigma: float,
    popsize: int,
) -> chex.Array:
    """Jittable Gaussian Sample Helper."""
    sigmas = jnp.sqrt(C) + sigma
    z = jax.random.normal(rng, (mean.shape[0], popsize))  # ~ N(0, I)
    y = jnp.diag(sigmas).dot(z)  # ~ N(0, C)
    y = jnp.swapaxes(y, 1, 0)
    x = mean + y  # ~ N(m, σ^2 C)
    return x


def standard_deviation_ratio(
    improvements: chex.Array,
    members_elite: chex.Array,
    mean: chex.Array,
    C: chex.Array,
) -> float:
    """SDR - relate dist. of improvements to mean in param space."""
    # Compute avg. member for candidates that improve fitness -> SDR
    x_avg_imp = jnp.sum(
        improvements[:, jnp.newaxis] * members_elite, axis=0
    ) / jnp.sum(improvements)
    conditioned_diff = (x_avg_imp - mean) / C
    sdr = jnp.max(jnp.abs(conditioned_diff))
    return sdr


def update_cov_amalgam(
    members_elite: chex.Array,
    C: chex.Array,
    mean: chex.Array,
    eta_sigma: float,
) -> chex.Array:
    """Iterative update of mean and mean shift based on elite and history."""
    S_bar = members_elite - mean
    # Univariate update to standard deviations
    new_C = (1 - eta_sigma) * C + eta_sigma * jnp.sum(
        S_bar ** 2, axis=0
    ) / members_elite.shape[0]
    return new_C
