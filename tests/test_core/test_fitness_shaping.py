"""Tests for fitness shaping functions."""

import jax
import jax.numpy as jnp
from evosax.core.fitness_shaping import (
    add_weight_decay,
    centered_rank_fitness_shaping_fn,
    identity_fitness_shaping_fn,
    normalize,
    normalize_fitness_shaping_fn,
    standardize_fitness_shaping_fn,
    weights_fitness_shaping_fn,
)


def test_normalize():
    """Test the normalize function."""
    # Test with 1D array
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize(a)
    assert jnp.allclose(normalized, jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

    # Test with 2D array
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    normalized = normalize(a, axis=1)
    expected = jnp.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    assert jnp.allclose(normalized, expected)

    # Test with custom min/max values
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize(a, minval=0.0, maxval=1.0)
    assert jnp.allclose(normalized, jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    # Test with all equal values
    a = jnp.array([2.0, 2.0, 2.0])
    normalized = normalize(a)
    assert jnp.allclose(normalized, jnp.array([0.0, 0.0, 0.0]))


def test_add_weight_decay():
    """Test the add_weight_decay function."""

    # Create a simple fitness shaping function for testing
    def test_fitness_fn(population, fitness, state, params):
        return fitness

    # Create a wrapped function with weight decay
    weight_decay = 0.1
    wrapped_fn = add_weight_decay(test_fitness_fn, weight_decay=weight_decay)

    # Test the wrapped function
    population = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fitness = jnp.array([0.5, 1.0])
    state = None
    params = None

    # Calculate expected result
    l2_penalty = weight_decay * jnp.sum(jnp.square(population), axis=-1)
    expected = fitness + l2_penalty

    result = wrapped_fn(population, fitness, state, params)
    assert jnp.allclose(result, expected)


def test_identity_fitness_shaping_fn():
    """Test the identity fitness shaping function."""
    population = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fitness = jnp.array([0.5, 1.0])
    state = None
    params = None

    result = identity_fitness_shaping_fn(population, fitness, state, params)
    assert jnp.array_equal(result, fitness)


def test_standardize_fitness_shaping_fn():
    """Test the standardize fitness shaping function."""
    population = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fitness = jnp.array([0.5, 1.5])
    state = None
    params = None

    result = standardize_fitness_shaping_fn(population, fitness, state, params)
    expected = jax.nn.standardize(fitness, epsilon=1e-8)
    assert jnp.allclose(result, expected)

    # Test with NaN values
    fitness_with_nan = jnp.array([0.5, jnp.nan])
    result = standardize_fitness_shaping_fn(population, fitness_with_nan, state, params)
    # The standardize function should handle NaNs correctly
    assert jnp.isnan(result[1])
    assert not jnp.isnan(result[0])


def test_normalize_fitness_shaping_fn():
    """Test the normalize fitness shaping function."""
    population = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fitness = jnp.array([0.5, 1.5])
    state = None
    params = None

    result = normalize_fitness_shaping_fn(population, fitness, state, params)
    expected = normalize(fitness)
    assert jnp.allclose(result, expected)


def test_centered_rank_fitness_shaping_fn():
    """Test the centered rank fitness shaping function."""
    population = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    fitness = jnp.array([3.0, 1.0, 2.0])  # Ranks would be [2, 0, 1]
    state = None
    params = None

    result = centered_rank_fitness_shaping_fn(population, fitness, state, params)
    # Expected: ranks / (n-1) - 0.5 = [2, 0, 1] / 2 - 0.5 = [1.0, 0.0, 0.5] - 0.5 = [0.5, -0.5, 0.0]
    expected = jnp.array([0.5, -0.5, 0.0])
    assert jnp.allclose(result, expected)


def test_weights_fitness_shaping_fn():
    """Test the weights fitness shaping function."""
    population = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    fitness = jnp.array([3.0, 1.0, 2.0])  # Ranks would be [2, 0, 1]
    state = None

    # Create params with weights
    class Params:
        def __init__(self, weights):
            self.weights = weights

    weights = jnp.array([0.1, 0.2, 0.3])
    params = Params(weights)

    result = weights_fitness_shaping_fn(population, fitness, state, params)
    # Expected: weights[ranks] = weights[[2, 0, 1]] = [0.3, 0.1, 0.2]
    expected = jnp.array([0.3, 0.1, 0.2])
    assert jnp.allclose(result, expected)
