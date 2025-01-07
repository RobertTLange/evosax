from typing import Union, Optional
import jax
import jax.numpy as jnp
import chex
from functools import partial


class FitnessShaper(object):
    def __init__(
        self,
        centered_rank: Union[bool, int] = False,
        z_score: Union[bool, int] = False,
        norm_range: Union[bool, int] = False,
        w_decay: float = 0.0,
        maximize: Union[bool, int] = False,
        fitness_trafo: Optional[str] = None,
    ):
        """JAX-compatible fitness shaping tool."""
        self.w_decay = w_decay
        self.maximize = bool(maximize)

        if fitness_trafo in ["centered_rank", "z_score", "norm_range", "raw"]:
            self.centered_rank = fitness_trafo == "centered_rank"
            self.z_score = fitness_trafo == "z_score"
            self.norm_range = fitness_trafo == "norm_range"
        else:
            self.centered_rank = bool(centered_rank)
            self.z_score = bool(z_score)
            self.norm_range = bool(norm_range)
        # Check that only single fitness shaping transformation is used
        num_options_on = self.centered_rank + self.z_score + self.norm_range
        assert (
            num_options_on < 2
        ), "Only use one fitness shaping transformation."

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, x: chex.Array, fitness: chex.Array) -> chex.Array:
        """Max objective trafo, rank shaping, z scoring & add weight decay."""
        if self.maximize:
            fitness = -1 * fitness

        # Apply wdecay before normalization - makes easier to tune
        # "Reduce" fitness based on L2 norm of parameters
        if self.w_decay > 0.0:
            l2_fit_red = self.w_decay * compute_l2_norm(x)
            fitness += l2_fit_red

        if self.centered_rank:
            fitness = centered_rank_trafo(fitness)

        if self.z_score:
            fitness = z_score_trafo(fitness)

        if self.norm_range:
            fitness = range_norm_trafo(fitness, -1.0, 1.0)

        return fitness


def z_score_trafo(arr: chex.Array) -> chex.Array:
    """Make fitness 'Gaussian' by substracting mean and dividing by std."""
    return (arr - jnp.nanmean(arr)) / (jnp.nanstd(arr) + 1e-10)


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
    return jnp.nanmean(x * x, axis=1)


def range_norm_trafo(
    arr: chex.Array, min_val: float = -1.0, max_val: float = 1.0
) -> chex.Array:
    """Map scores into a min/max range."""
    arr = jnp.clip(arr, -1e10, 1e10)
    normalized_arr = (max_val - min_val) * (arr - jnp.nanmin(arr)) / (
        jnp.nanmax(arr) - jnp.nanmin(arr) + 1e-10
    ) + min_val
    return normalized_arr
