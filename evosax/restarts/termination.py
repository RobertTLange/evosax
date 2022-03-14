import chex
import jax.numpy as jnp


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
    """Termination criterion specific to the CMA-ES strategy."""
    return False


# def cma_criterion(
#     fitness: chex.Array, state: chex.ArrayTree, params: chex.ArrayTree
# ):
#     """Check whether to terminate CMA-ES loop."""
#     dC = jnp.diag(state["C"])
#     C, B, D = eigen_decomposition(state["C"], state["B"], state["D"])

#     # Stop if std of normal distrib is smaller than tolx in all coordinates
#     # and pc is smaller than tolx in all components.
#     if jnp.all(state["sigma"] * dC < params["tol_x"]) and np.all(
#         state["sigma"] * state["p_c"] < params["tol_x"]
#     ):
#         print("TERMINATE ----> Convergence/Search variance too small")
#         return True

#     # Stop if detecting divergent behavior.
#     if state["sigma"] * jnp.max(D) > params["tol_x_up"]:
#         print("TERMINATE ----> Stepsize sigma exploded")
#         return True

#     # No effect coordinates: stop if adding 0.2-standard deviations
#     # in any single coordinate does not change m.
#     if jnp.any(
#         state["mean"] == state["mean"] + (0.2 * state["sigma"] * jnp.sqrt(dC))
#     ):
#         print("TERMINATE ----> No effect when adding std to mean")
#         return True

#     # No effect axis: stop if adding 0.1-standard deviation vector in
#     # any principal axis direction of C does not change m.
#     if jnp.all(
#         state["mean"] == state["mean"] + (0.1 * state["sigma"] * D[0] * B[:, 0])
#     ):
#         print("TERMINATE ----> No effect when adding std to mean")
#         return True

#     # Stop if the condition number of the covariance matrix exceeds 1e14.
#     condition_cov = jnp.max(D) / jnp.min(D)
#     if condition_cov > params["tol_condition_C"]:
#         print("TERMINATE ----> C condition number exploded")
#         return True
#     return False
