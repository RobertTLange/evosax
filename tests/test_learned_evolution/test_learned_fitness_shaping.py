"""Tests for feature transforms used by pretrained learned optimizers."""

import jax.numpy as jnp
from evosax.learned_evolution.fitness_shaping import normalize, standardize
from evosax.learned_evolution.lga_tools import normalize_lga_mutation_strength


def test_standardize_uses_population_statistics():
    """Mutation strengths are compared across the population."""
    mutation_strengths = jnp.array([[1.0], [2.0]])

    result = standardize(mutation_strengths)

    assert jnp.allclose(result, jnp.array([[-1.0], [1.0]]))


def test_standardize_equal_population_is_neutral():
    """Equal mutation strengths must remain finite and neutral."""
    mutation_strengths = jnp.full((8, 1), 2.1803024)

    result = standardize(mutation_strengths)

    assert jnp.array_equal(result, jnp.zeros_like(mutation_strengths))


def test_normalize_uses_requested_range():
    """Later learned checkpoints use conventional range normalization."""
    mutation_strengths = jnp.array([[1.0], [2.0]])

    result = normalize(mutation_strengths, min_val=-0.5, max_val=0.5)

    assert jnp.allclose(result, jnp.array([[-0.5], [0.5]]))


def test_normalize_equal_values_use_lower_bound():
    """A constant feature has no position within the requested range."""
    mutation_strengths = jnp.ones((8, 1))

    result = normalize(mutation_strengths)

    assert jnp.array_equal(result, -jnp.ones_like(mutation_strengths))


def test_lga_mutation_normalization_preserves_checkpoint_range():
    """The 2023 LGA checkpoints retain their original mutation encoding."""
    mutation_strengths = jnp.array([[1.0], [2.0]])

    result = normalize_lga_mutation_strength(mutation_strengths)

    assert jnp.allclose(result, jnp.array([[1.0], [3.0]]))


def test_lga_equal_mutation_strength_preserves_checkpoint_baseline():
    """Equal mutation strengths retain LGA's checkpoint baseline."""
    mutation_strengths = jnp.ones((8, 1))

    result = normalize_lga_mutation_strength(mutation_strengths)

    assert jnp.array_equal(result, jnp.ones_like(mutation_strengths))
