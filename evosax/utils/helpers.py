import jax
import jax.numpy as jnp
from jax import flatten_util

from ..types import Fitness, Population, Solution


def get_ravel_fn(solution: Solution):
    def ravel_solution(solution):
        flat, _ = flatten_util.ravel_pytree(solution)
        return flat

    _, unravel_solution = flatten_util.ravel_pytree(solution)

    return ravel_solution, unravel_solution


def get_best_fitness_member(
    x: Population, fitness: Fitness, state, maximize: bool = False
) -> tuple[jax.Array, float]:
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
