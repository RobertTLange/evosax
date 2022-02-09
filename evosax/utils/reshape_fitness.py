import jax
import jax.numpy as jnp
import chex
from functools import partial


class FitnessShaper(object):
    def __init__(
        self,
        rank_fitness: bool = False,
        z_score_fitness: bool = False,
        weight_decay: float = 0.0,
        maximize_objective: bool = False,
    ):
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        self.z_score_fitness = z_score_fitness
        self.maximize_objective = maximize_objective

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x: chex.Array, fitness: chex.Array):
        """Max objective trafo, rank shaping, z scoring and add weight decay."""
        fitness = jax.lax.select(self.maximize_objective, -1 * fitness, fitness)
        fitness = jax.lax.select(
            self.rank_fitness, compute_centered_ranks(fitness), fitness
        )
        fitness = jax.lax.select(
            self.z_score_fitness, z_score_fitness(fitness), fitness
        )
        # "Reduce" fitness based on L2 norm of parameters
        l2_fitness_reduction = self.weight_decay * compute_weight_norm(x)
        return fitness + l2_fitness_reduction


def z_score_fitness(fitness: chex.Array):
    """Make fitness 'Gaussian' by substracting mean and dividing by std."""
    return (fitness - jnp.mean(fitness)) / jnp.std(1e-6 + fitness)


def compute_ranks(fitness: chex.Array):
    """Return ranks in [0, len(fitness))."""
    ranks = jnp.zeros(len(fitness))
    ranks = ranks.at[fitness.argsort()].set(jnp.arange(len(fitness)))
    return ranks


def compute_centered_ranks(fitness: chex.Array):
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!)."""
    y = compute_ranks(fitness)
    y /= fitness.size - 1
    y -= 0.5
    return y


def compute_weight_norm(x: chex.Array):
    """Compute L2-norm of weights. Assumes x to be (popsize, num_dims)."""
    return jnp.mean(x * x, axis=1)
