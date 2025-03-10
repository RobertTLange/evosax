"""Tests for kernel functions."""

import jax
import jax.numpy as jnp
from evosax.core.kernel import kernel_rbf


def test_kernel_rbf():
    """Test the radial basis function kernel."""
    # Create sample inputs
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.0, 2.0, 3.0])

    # Create a simple params class for testing
    class Params:
        def __init__(self, kernel_std):
            self.kernel_std = kernel_std

    # Test with identical vectors
    params = Params(kernel_std=1.0)
    result = kernel_rbf(x, y, params)
    # When x and y are identical, the kernel should return 1.0
    assert jnp.isclose(result, 1.0)

    # Test with different vectors
    y = jnp.array([2.0, 3.0, 4.0])
    result = kernel_rbf(x, y, params)
    # Calculate expected result manually
    dist_sq = jnp.sum(jnp.square((x - y) / params.kernel_std))
    expected = jnp.exp(-0.5 * dist_sq)
    assert jnp.isclose(result, expected)

    # Test with different kernel_std
    params = Params(kernel_std=2.0)
    result = kernel_rbf(x, y, params)
    # Calculate expected result with new kernel_std
    dist_sq = jnp.sum(jnp.square((x - y) / params.kernel_std))
    expected = jnp.exp(-0.5 * dist_sq)
    assert jnp.isclose(result, expected)

    # Test with batch of vectors
    x_batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_batch = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    # Apply kernel to each pair in the batch
    results = jax.vmap(lambda x, y: kernel_rbf(x, y, params))(x_batch, y_batch)

    # Calculate expected results manually
    expected_results = jnp.array(
        [
            jnp.exp(
                -0.5
                * jnp.sum(jnp.square((x_batch[0] - y_batch[0]) / params.kernel_std))
            ),
            jnp.exp(
                -0.5
                * jnp.sum(jnp.square((x_batch[1] - y_batch[1]) / params.kernel_std))
            ),
        ]
    )

    assert jnp.allclose(results, expected_results)
