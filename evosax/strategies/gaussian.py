import jax
import jax.numpy as jnp
from jax import jit
import functools


def init_strategy(mean_init, sigma_init, population_size, mu,
                  split_params=False):
    ''' Initialize evolutionary strategy & learning rates. '''
    n_dim = mean_init.shape[0]
    weights = jnp.zeros(population_size)
    # Only parents have positive weight - equal weighting!
    weights = jax.ops.index_update(weights, jax.ops.index[:mu], 1/mu)
    memory = {"sigma": sigma_init,
              "mean": mean_init,
              "generation": 0}

    # Decide whether to split params in static and dynamic hyperparams
    if not split_params:
        params = {"pop_size": population_size,
                  "n_dim": n_dim,
                  "weights": weights,
                  "c_m": 1,  # Learning rate for population mean
                  "c_sigma": 0.,   # Learning rate for population std
                  "tol_fun": 1e-12,
                  "min_generations": 10}
        return params, memory
    else:
        ask_params = {"pop_size": population_size,
                      "n_dim": n_dim}
        tell_params = {"c_m": 1,
                       "c_sigma": 0.,
                       "weights": weights}
        terminal_params = {"tol_fun": 1e-12}
        return ask_params, tell_params, terminal_params, memory



def ask_gaussian_strategy(rng, params, memory):
    """ Propose params to evaluate next. Sample from isotropic Gaussian. """
    x = sample(rng, memory, params["n_dim"], params["pop_size"])
    return x, memory


@functools.partial(jax.jit, static_argnums=(2, 3))
def sample(rng, memory, n_dim, pop_size):
    """ Jittable Gaussian Sample Helper. """
    z = jax.random.normal(rng, (pop_size, n_dim)) # ~ N(0, I)
    x = memory["mean"] + memory["sigma"] * z    # ~ N(m, σ^2 I)
    return x


def tell_gaussian_strategy(x, fitness, params, memory):
    """ Update the surrogate ES model. """
    memory["generation"] = memory["generation"] + 1
    # Sort new results, extract elite, store best performer
    concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
    sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
    # Update mean, isotropic/anisotropic paths, covariance, stepsize
    mean, y_k = update_mean(sorted_solutions, params, memory)
    sigma = update_sigma(y_k, params, memory)
    memory["mean"], memory["sigma"] = mean, sigma
    return memory


# Jitted version of Gaussian ask and tell interface
ask = ask_gaussian_strategy
tell = jit(tell_gaussian_strategy)


def update_mean(sorted_solutions, params, memory):
    """ Update mean of strategy. """
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = (x_k - memory["mean"])
    y_w = jnp.sum(y_k.T * params["weights"], axis=1)
    mean = memory["mean"] + params["c_m"] * y_w
    return mean, y_k


def update_sigma(y_k, params, memory):
    """ Update stepsize sigma. """
    sigma_est = jnp.sqrt(jnp.sum((y_k.T**2 * params["weights"]), axis=1))
    sigma = ((1-params["c_sigma"])*memory["sigma"]
              + params["c_sigma"]*sigma_est)
    return sigma


def check_termination(values, params, memory):
    """ Check whether to terminate simple Gaussian search loop. """
    # Stop if generation fct values of recent generation is below thresh.
    if (memory["generation"] > params["min_generations"]
        and jnp.max(values) - jnp.min(values) < params["tol_fun"]):
        print("TERMINATE ----> Convergence/No progress in objective")
        return True
    return False
