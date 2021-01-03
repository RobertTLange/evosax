import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.scipy.stats.multivariate_normal import logpdf
import functools
import optax


def init_strategy(lrate, search_params_init,
                  sigma, population_size):
    ''' Initialize evolutionary strategy & learning rates. '''
    n_dim = search_params_init.shape[0]
    params = {"pop_size": population_size,
              "n_dim": n_dim,
              "lrate": lrate,
              "tol_fun": 1e-12,
              "min_generations": 10}
    memory = {"search_params": search_params_init,
              "sigma": sigma,
              "generation": 0}
    return params, memory


def ask_open_nes_strategy(rng, params, memory):
    """ Propose params to evaluate next. Sample from Multivariate Gaussian. """
    z = antithetic_sample(rng, memory, params["n_dim"], params["pop_size"])  # ~ N(0, I)
    x = memory["search_params"] + memory["sigma"]*z
    return x, memory


@functools.partial(jax.jit, static_argnums=(2, 3))
def antithetic_sample(rng, memory, n_dim, pop_size):
    """ Jittable Gaussian Sample Helper. """
    z_plus = jax.random.multivariate_normal(rng, jnp.zeros(n_dim), # ~ N(0, I)
                                            jnp.eye(n_dim), (int(pop_size/2),))
    z = jnp.concatenate([z_plus, -1.*z_plus])
    return z


def tell_open_nes_strategy(x, fitness, params, memory, opt_state):
    """ Update the surrogate ES model. """
    memory["generation"] = memory["generation"] + 1

    # Get REINFORCE-style gradient for each sample
    noise = (x - memory["search_params"])/memory["sigma"]
    nes_grads = (1./(params["pop_size"]*memory["sigma"])
                 * jnp.dot(noise.T, fitness))

    # Natural grad update using optax API - redefine adam opt without init!
    opt = optax.adam(params["lrate"])
    updates, opt_state = opt.update(nes_grads, opt_state)
    memory["search_params"] = optax.apply_updates(memory["search_params"], updates)
    return memory, opt_state


# Jitted version of CMA-ES ask and tell interface
ask = ask_open_nes_strategy
tell = jit(tell_open_nes_strategy)
