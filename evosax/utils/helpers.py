import chex
import jax
import jax.numpy as jnp
from jax import flatten_util


def get_ravel_fn(pholder_params: chex.ArrayTree):
    def ravel_params(params):
        flat, _ = flatten_util.ravel_pytree(params)
        return flat

    _, unravel_params = flatten_util.ravel_pytree(pholder_params)

    return ravel_params, unravel_params


def get_best_fitness_member(
    x: chex.Array, fitness: chex.Array, state, maximize: bool = False
) -> tuple[chex.Array, float]:
    """Check if fitness improved & replace in ES state."""
    fitness_min = jax.lax.select(maximize, -1 * fitness, fitness)
    max_and_later = maximize and state.generation_counter > 0
    best_fit_min = jax.lax.select(
        max_and_later, -1 * state.best_fitness, state.best_fitness
    )
    best_in_gen = jnp.argmin(fitness_min)
    best_in_gen_fitness, best_in_gen_member = (
        fitness_min[best_in_gen],
        x[best_in_gen],
    )
    replace_best = best_in_gen_fitness < best_fit_min
    best_fitness = jax.lax.select(replace_best, best_in_gen_fitness, best_fit_min)
    best_member = jax.lax.select(replace_best, best_in_gen_member, state.best_member)
    best_fitness = jax.lax.select(maximize, -1 * best_fitness, best_fitness)
    return best_member, best_fitness
