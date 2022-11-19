import jax
import jax.numpy as jnp
import chex
from functools import partial


class FitnessShaper(object):
    def __init__(
        self,
        centered_rank: bool = False,
        z_score: bool = False,
        norm_range: bool = False,
        w_decay: float = 0.0,
        maximize: bool = False,
    ):
        """JAX-compatible fitness shaping tool."""
        self.w_decay = w_decay
        self.centered_rank = bool(centered_rank)
        self.z_score = bool(z_score)
        self.norm_range = bool(norm_range)
        self.maximize = bool(maximize)
        # TODO: Add assert statement to check that only one condition is met

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x: chex.Array, fitness: chex.Array) -> chex.Array:
        """Max objective trafo, rank shaping, z scoring & add weight decay."""
        fitness = jax.lax.select(self.maximize, -1 * fitness, fitness)
        fitness = jax.lax.select(
            self.centered_rank, centered_rank_trafo(fitness), fitness
        )
        fitness = jax.lax.select(self.z_score, z_score_trafo(fitness), fitness)
        fitness = jax.lax.select(
            self.norm_range, range_norm_trafo(fitness, -1.0, 1.0), fitness
        )
        # "Reduce" fitness based on L2 norm of parameters
        l2_fit_red = self.w_decay * compute_l2_norm(x)
        l2_fit_red = jax.lax.select(self.maximize, -1 * l2_fit_red, l2_fit_red)
        return fitness + l2_fit_red


def z_score_trafo(arr: chex.Array) -> chex.Array:
    """Make fitness 'Gaussian' by substracting mean and dividing by std."""
    return (arr - jnp.mean(arr)) / (jnp.std(arr) + 1e-10)


def compute_ranks(fitness: chex.Array) -> chex.Array:
    """Return fitness ranks in [0, len(fitness))."""
    ranks = jnp.zeros(len(fitness))
    ranks = ranks.at[fitness.argsort()].set(jnp.arange(len(fitness)))
    return ranks


def centered_rank_trafo(fitness: chex.Array) -> chex.Array:
    """Return ~ -0.5 to 0.5 centered ranks (best to worst - min!)."""
    y = compute_ranks(fitness)
    y /= fitness.size - 1
    return y - 0.5


def compute_l2_norm(x: chex.Array) -> chex.Array:
    """Compute L2-norm of x_i. Assumes x to have shape (popsize, num_dims)."""
    return jnp.mean(x * x, axis=1)


def range_norm_trafo(
    arr: chex.Array, min_val: float = -1.0, max_val: float = 1.0
) -> chex.Array:
    """Map scores into a min/max range."""
    arr = jnp.clip(arr, -1e10, 1e10)
    normalized_arr = (
        2
        * max_val
        * (arr - jnp.nanmin(arr))
        / (jnp.nanmax(arr) - jnp.nanmin(arr) + 1e-10)
        - min_val
    )
    return normalized_arr
