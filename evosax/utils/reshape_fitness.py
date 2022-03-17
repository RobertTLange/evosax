import jax
import jax.numpy as jnp
import chex
from functools import partial


class FitnessShaper(object):
    def __init__(
        self,
        centered_rank: bool = False,
        z_score: bool = False,
        w_decay: float = 0.0,
        maximize: bool = False,
    ):
        """JAX-compatible fitness shaping tool."""
        self.w_decay = w_decay
        self.centered_rank = bool(centered_rank)
        self.z_score = bool(z_score)
        self.maximize = bool(maximize)

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x: chex.Array, fitness: chex.Array) -> chex.Array:
        """Max objective trafo, rank shaping, z scoring & add weight decay."""
        fitness = jax.lax.select(self.maximize, -1 * fitness, fitness)
        fitness = jax.lax.select(
            self.centered_rank, compute_centered_ranks(fitness), fitness
        )
        fitness = jax.lax.select(
            self.z_score, z_score_fitness(fitness), fitness
        )
        # "Reduce" fitness based on L2 norm of parameters
        l2_fit_red = self.w_decay * compute_weight_norm(x)
        l2_fit_red = jax.lax.select(self.maximize, -1 * l2_fit_red, l2_fit_red)
        return fitness + l2_fit_red


def z_score_fitness(fitness: chex.Array) -> chex.Array:
    """Make fitness 'Gaussian' by substracting mean and dividing by std."""
    return (fitness - jnp.mean(fitness)) / jnp.std(1e-05 + fitness)


def compute_ranks(fitness: chex.Array) -> chex.Array:
    """Return fitness ranks in [0, len(fitness))."""
    ranks = jnp.zeros(len(fitness))
    ranks = ranks.at[fitness.argsort()].set(jnp.arange(len(fitness)))
    return ranks


def compute_centered_ranks(fitness: chex.Array) -> chex.Array:
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!)."""
    y = compute_ranks(fitness)
    y /= fitness.size - 1
    return y - 0.5


def compute_weight_norm(x: chex.Array) -> chex.Array:
    """Compute L2-norm of x_i. Assumes x to have shape (popsize, num_dims)."""
    return jnp.mean(x * x, axis=1)
