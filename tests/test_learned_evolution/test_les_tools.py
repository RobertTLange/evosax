"""Tests for learned-evolution utility functions."""

import jax.numpy as jnp
from evosax.learned_evolution.les_tools import norm_diff_best


def test_norm_diff_best_normalizes_fitness_gap() -> None:
    fitness = jnp.array([2.0, 3.0, 5.0])

    result = norm_diff_best(fitness, best_fitness=1.0)

    assert jnp.allclose(result, jnp.array([1 / 3, 2 / 3, 1.0]))
