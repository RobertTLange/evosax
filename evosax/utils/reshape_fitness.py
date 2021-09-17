import jax
import jax.numpy as jnp
from functools import partial


class FitnessShaper(object):
    def __init__(self,
                 rank_fitness: bool = False,
                 z_score_fitness: bool = False,
                 weight_decay: float = 0.0):
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        self.z_score_fitness = z_score_fitness

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x, fitness):
        """ Apply weight decay and rank shaping. """
        fitness = jax.lax.select(self.rank_fitness,
                                 compute_centered_ranks(fitness),
                                 fitness)
        fitness = jax.lax.select(self.z_score_fitness,
                                 z_score_fitness(fitness),
                                 fitness)
        # "Reduce" fitness based on L2 norm of parameters
        l2_fitness_reduction = - self.weight_decay * compute_weight_norm(x)
        return fitness + l2_fitness_reduction


def z_score_fitness(fitness):
    """ Make fitness 'Gaussian' by substracting mean and dividing by std. """
    return (fitness - jnp.mean(fitness)) / jnp.std(fitness)


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
