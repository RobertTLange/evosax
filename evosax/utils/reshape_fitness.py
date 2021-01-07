import jax.numpy as jnp
from jax import jit


@jit
def z_score_fitness(fitness):
    """ Make fitness 'Gaussian' by substracting mean and dividing by std. """
    return (fitness - jnp.mean(fitness)) / jnp.std(fitness)


@jit
def rank_shaped_fitness(x, fitness, fit_range=[-0.5, 0.5]):
    """ REINFORCE weights scaled between [-1, 1] - variance reduction. """
    # Sort new results, extract elite, store best performer
    concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
    idx_sort = concat_p_f[:, 0].argsort()
    sorted_solutions = x[idx_sort]
    # Return linear spaced weights between [-1, 1]
    shaped_fitness = jnp.linspace(fit_range[0],
                                  fit_range[1], x.shape[0])
    return sorted_solutions, shaped_fitness
