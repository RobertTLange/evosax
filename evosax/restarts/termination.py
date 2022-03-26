import chex
import jax.numpy as jnp
from ..utils.eigen_decomp import full_eigen_decomp


def min_gen_criterion(
    fitness: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
) -> bool:
    """Allow stopping if minimal number of generations has passed."""
    min_gen_passed = state["gen_counter"] >= params["min_num_gens"]
    return min_gen_passed


def spread_criterion(
    fitness: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
) -> bool:
    """Stop if min/max fitness spread of recent generation is below thresh."""
    fit_var_too_low = (
        jnp.max(fitness) - jnp.min(fitness) < params["min_fitness_spread"]
    )
    return fit_var_too_low


def cma_criterion(
    fitness: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
) -> bool:
    """Termination criterion specific to CMA-ES strategy. Default tolerances:
    tol_x - 1e-12 * sigma
    tol_x_up - 1e4
    tol_condition_C - 1e14
    """
    cma_term = 0
    dC = jnp.diag(state["C"])
    # Note: Criterion requires full covariance matrix for decomposition!
    C, B, D = full_eigen_decomp(state["C"], state["B"], state["D"])

    # Stop if std of normal distrib is smaller than tolx in all coordinates
    # and pc is smaller than tolx in all components.
    cond_s_1 = jnp.all(state["sigma"] * dC < params["tol_x"])
    cond_s_2 = jnp.all(state["sigma"] * state["p_c"] < params["tol_x"])
    cma_term += jnp.logical_and(cond_s_1, cond_s_2)

    # Stop if detecting divergent behavior -> Stepsize sigma exploded.
    cma_term += state["sigma"] * jnp.max(D) > params["tol_x_up"]

    # No effect coordinates: stop if adding 0.2-standard deviations
    # in any single coordinate does not change mean.
    cond_no_coord_change = jnp.any(
        state["mean"] == state["mean"] + (0.2 * state["sigma"] * jnp.sqrt(dC))
    )
    cma_term += cond_no_coord_change

    # No effect axis: stop if adding 0.1-standard deviation vector in
    # any principal axis direction of C does not change m.
    cond_no_axis_change = jnp.all(
        state["mean"] == state["mean"] + (0.1 * state["sigma"] * D[0] * B[:, 0])
    )
    cma_term += cond_no_axis_change

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    cond_condition_cov = jnp.max(D) / jnp.min(D) > params["tol_condition_C"]
    cma_term += cond_condition_cov
    return cma_term > 0


def amalgam_criterion(
    fitness: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
) -> bool:
    """Termination criterion for iAMaLGaM algorithm (Bosman et al. 2013)"""
    return state["c_mult"] < 1e-10
