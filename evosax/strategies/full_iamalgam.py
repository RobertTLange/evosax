import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class Full_iAMaLGaM(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.35):
        """(Iterative) AMaLGaM (Bosman et al., 2013) - Full Covariance
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
        self.strategy_name = "Full_iAMaLGaM"

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
            "C": jnp.eye(self.num_dims),
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
    S = C + sigma ** 2 * jnp.eye(C.shape[0])
    candidates = jax.random.multivariate_normal(
        rng, mean, S, (popsize,)
    )  # ~ N(m, S) - shape: (popsize, num_dims)
    return candidates


def anticipated_mean_shift(
    rng: chex.PRNGKey,
    x: chex.Array,
    ams_popsize: int,
    delta_ams: float,
    c_mult: float,
    mean_shift: chex.Array,
) -> chex.Array:
    """AMS - move part of pop further into dir of anticipated improvement."""
    indices = jnp.arange(x.shape[0])
    sample_idx = jax.random.choice(rng, indices, (ams_popsize,), replace=False)
    x_ams_new = x[sample_idx] + c_mult * delta_ams * mean_shift
    x_ams = x.at[sample_idx].set(x_ams_new)
    return x_ams


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
    # Expensive! Can we somehow reuse this in sampling step?
    L = jax.scipy.linalg.cholesky(C)
    conditioned_diff = jnp.linalg.inv(L) @ (x_avg_imp - mean)
    sdr = jnp.max(jnp.abs(conditioned_diff))
    return sdr


def adaptive_variance_scaling(
    any_improvement: bool,
    sdr: float,
    c_mult: float,
    nis_counter: int,
    theta_sdr: float,
    eta_avs_inc: float,
    eta_avs_dec: float,
    nis_max_gens: int,
) -> Tuple[float, int]:
    """AVS - adaptively rescale covariance depending on SDR."""
    # Case 1: If improvement in best fitness -> SDR increase c_mult! [L14-19]
    new_nis_counter = jax.lax.select(any_improvement, 0, nis_counter)
    reset_criterion = jnp.logical_and(any_improvement, c_mult < 1)
    c_mult = jax.lax.select(reset_criterion, 1.0, c_mult)
    sdr_criterion = jnp.logical_and(any_improvement, sdr > theta_sdr)
    c_mult_inc = jax.lax.select(sdr_criterion, eta_avs_inc * c_mult, c_mult)

    # Case 2: If  no improvement in best fitness -> Decrease c_mult! [L21-24]
    nis_dec_criterion = jnp.logical_and(1 - any_improvement, c_mult <= 1)
    new_nis_counter = jax.lax.select(
        nis_dec_criterion, nis_counter + 1, new_nis_counter
    )
    dec_criterion = jnp.logical_and(
        1 - any_improvement,
        jnp.logical_or(c_mult > 1, new_nis_counter >= nis_max_gens),
    )
    c_mult_dec = jax.lax.select(dec_criterion, eta_avs_dec * c_mult, c_mult)
    c_dec_criterion = jnp.logical_and(
        1 - any_improvement, new_nis_counter < nis_max_gens
    )
    c_dec_reset_criterion = jnp.logical_and(c_dec_criterion, c_mult_dec < 1)
    c_mult_dec = jax.lax.select(c_dec_reset_criterion, 1.0, c_mult_dec)

    # Select new multiplier based on case at hand
    new_c_mult = jax.lax.select(any_improvement, c_mult_inc, c_mult_dec)
    return new_c_mult, new_nis_counter


def update_mean_amalgam(
    members_elite: chex.Array,
    mean: chex.Array,
    mean_shift: chex.Array,
    eta_shift: float,
) -> Tuple[chex.Array, chex.Array]:
    """Iterative update of mean and mean shift based on elite and history."""
    new_mean = jnp.mean(members_elite, axis=0)
    new_mean_shift = (1 - eta_shift) * mean_shift + eta_shift * (
        new_mean - mean
    )
    return new_mean, new_mean_shift


def update_cov_amalgam(
    members_elite: chex.Array,
    C: chex.Array,
    mean: chex.Array,
    eta_sigma: float,
) -> chex.Array:
    """Iterative update of mean and mean shift based on elite and history."""
    S_bar = members_elite - mean
    new_C = (1 - eta_sigma) * C + eta_sigma * (
        S_bar.T @ S_bar
    ) / members_elite.shape[0]
    return new_C
