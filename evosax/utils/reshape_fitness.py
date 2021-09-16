import jax
import jax.numpy as jnp


@jax.jit
def z_score_fitness(fitness):
    """ Make fitness 'Gaussian' by substracting mean and dividing by std. """
    return (fitness - jnp.mean(fitness)) / jnp.std(fitness)


@jax.jit
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


class FitnessShaper(object):
    def __init__(self, rank_fitness: bool = True,
                 weight_decay: float = 0.01):
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x, fitness):
        """ Apply weight decay and rank shaping. """
        fitness_trafo = jax.lax.select(self.rank_fitness,
                                       compute_centered_ranks(fitness),
                                       fitness)
        l2_fitness_reduction = - self.weight_decay * compute_weight_norm(x)
        return fitness_trafo + l2_fitness_reduction


def compute_ranks(fitness):
    """ Return ranks in [0, len(fitness)). """
    ranks = jnp.arange(len(fitness))[fitness.argsort()]
    return ranks


def compute_centered_ranks(fitness):
    """ Return ~ -0.5 to 0.5 centered ranks. """
    y = compute_ranks(fitness)
    y /= (fitness.size - 1)
    y -= .5
    return y


def compute_weight_norm(x):
    """ Compute L2-norm of weights. Assumes x to be (popsize, num_dims). """
    return jnp.mean(x * x, axis=1)
