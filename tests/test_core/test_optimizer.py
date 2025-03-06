"""Tests for optimizer functions."""

import jax.numpy as jnp
import optax
from evosax.core.optimizer import ScaleByClipUpState, clipup, scale_by_clipup


def test_scale_by_clipup():
    """Test the scale_by_clipup transformation."""
    # Create test parameters
    params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array(0.5)}
    updates = {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array(0.05)}

    # Test initialization
    max_velocity_ratio = 2.0
    momentum = 0.9
    transform = scale_by_clipup(
        max_velocity_ratio=max_velocity_ratio, momentum=momentum
    )
    state = transform.init(params)

    # Check initial state
    assert isinstance(state, ScaleByClipUpState)
    assert jnp.array_equal(state.velocity["w"], jnp.zeros_like(params["w"]))
    assert jnp.array_equal(state.velocity["b"], jnp.zeros_like(params["b"]))
    assert state.count == 0

    # Test update function - first update (no clipping needed)
    new_updates, new_state = transform.update(updates, state)

    # Check that velocity is updated with momentum
    expected_velocity_w = momentum * state.velocity["w"] + updates["w"]
    expected_velocity_b = momentum * state.velocity["b"] + updates["b"]
    assert jnp.allclose(new_state.velocity["w"], expected_velocity_w)
    assert jnp.allclose(new_state.velocity["b"], expected_velocity_b)
    assert new_state.count == 1

    # Test update function with velocity exceeding max_velocity_ratio
    large_updates = {"w": jnp.array([10.0, 20.0, 30.0]), "b": jnp.array(5.0)}
    new_updates, new_state = transform.update(large_updates, state)

    # Calculate expected results
    expected_velocity_w = momentum * state.velocity["w"] + large_updates["w"]
    expected_velocity_b = momentum * state.velocity["b"] + large_updates["b"]

    # Calculate the norm and scaling factor
    velocity_flat = jnp.concatenate(
        [expected_velocity_w.flatten(), jnp.array([expected_velocity_b])]
    )
    velocity_norm = jnp.sqrt(jnp.sum(jnp.square(velocity_flat)))
    scale = max_velocity_ratio / velocity_norm  # Assuming norm > max_velocity_ratio

    # Apply scaling
    expected_clipped_velocity_w = expected_velocity_w * scale
    expected_clipped_velocity_b = expected_velocity_b * scale

    # Check that velocity is clipped correctly
    assert jnp.allclose(new_state.velocity["w"], expected_clipped_velocity_w)
    assert jnp.allclose(new_state.velocity["b"], expected_clipped_velocity_b)

    # Verify that the norm of the clipped velocity is <= max_velocity_ratio
    clipped_velocity_flat = jnp.concatenate(
        [new_state.velocity["w"].flatten(), jnp.array([new_state.velocity["b"]])]
    )
    clipped_velocity_norm = jnp.sqrt(jnp.sum(jnp.square(clipped_velocity_flat)))
    assert (
        clipped_velocity_norm <= max_velocity_ratio + 1e-6
    )  # Allow for numerical precision


def test_clipup():
    """Test the clipup optimizer."""
    # Create test parameters
    params = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array(0.5)}
    updates = {"w": jnp.array([0.1, 0.2, 0.3]), "b": jnp.array(0.05)}

    # Test initialization
    learning_rate = 0.01
    max_velocity = 0.1
    momentum = 0.9
    eps = 1e-8

    optimizer = clipup(
        learning_rate=learning_rate,
        max_velocity=max_velocity,
        momentum=momentum,
        eps=eps,
    )

    state = optimizer.init(params)

    # Test update function
    new_updates, new_state = optimizer.update(updates, state)

    # Test with a large update to ensure clipping works
    large_updates = {"w": jnp.array([10.0, 20.0, 30.0]), "b": jnp.array(5.0)}
    new_updates, new_state = optimizer.update(large_updates, state)

    # Calculate the norm of the final updates
    updates_flat = jnp.concatenate(
        [new_updates["w"].flatten(), jnp.array([new_updates["b"]])]
    )
    updates_norm = jnp.sqrt(jnp.sum(jnp.square(updates_flat)))

    # The norm should be less than or equal to max_velocity
    assert updates_norm <= max_velocity + 1e-6  # Allow for numerical precision
