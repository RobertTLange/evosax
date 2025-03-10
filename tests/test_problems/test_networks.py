"""Tests for neural network architectures in evosax.problems.networks."""

import jax
import jax.numpy as jnp
from evosax.problems import (
    CNN,
    MLP,
    categorical_output_fn,
    identity_output_fn,
    tanh_output_fn,
)
from flax import linen as nn


def test_mlp():
    """Test MLP network with default settings."""
    key = jax.random.key(0)
    network = MLP(
        layer_sizes=(64, 64, 1),
        activation=nn.tanh,
        output_fn=identity_output_fn,
    )
    solution = jnp.zeros((4,))
    params = network.init(
        key,
        x=solution,
        key=key,
    )
    out = jax.jit(network.apply)(params, solution, key)
    assert out.shape == (1,)


def test_mlp_with_layer_norm():
    """Test MLP network with layer normalization."""
    key = jax.random.key(0)
    network = MLP(
        layer_sizes=(64, 64, 1),
        activation=nn.relu,
        output_fn=identity_output_fn,
        layer_norm=True,
    )
    solution = jnp.zeros((4,))
    params = network.init(
        key,
        x=solution,
        key=key,
    )
    out = jax.jit(network.apply)(params, solution, key)
    assert out.shape == (1,)


def test_cnn():
    """Test CNN network with default settings."""
    key = jax.random.key(0)
    network = CNN(
        num_filters=(16, 8),
        kernel_sizes=((3, 3), (5, 5)),
        strides=((1, 1), (1, 1)),
        mlp_layer_sizes=(16, 10),
        activation=nn.relu,
        output_fn=identity_output_fn,
    )
    solution = jnp.zeros((1, 28, 28, 1))
    params = network.init(
        key,
        x=solution,
        key=key,
    )
    out = jax.jit(network.apply)(params, solution, key)
    assert out.shape == (1, 10)


def test_identity_output_fn():
    """Test identity output function."""
    x = jnp.array([1.0, 2.0, 3.0])
    result = identity_output_fn(x)
    assert jnp.array_equal(result, x)


def test_categorical_output_fn():
    """Test categorical output function."""
    key = jax.random.key(0)
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    result = categorical_output_fn(logits, key)
    # With these logits, we expect the categorical function to select the highest value
    assert result.shape == (2,)
    assert result[0] == 0  # First row has highest value in first position
    assert result[1] == 1  # Second row has highest value in second position


def test_tanh_output_fn():
    """Test tanh output function."""
    x = jnp.array([0.0, 1.0, -1.0])
    result = tanh_output_fn(x)
    expected = jnp.array([0.0, jnp.tanh(1.0), jnp.tanh(-1.0)])
    assert jnp.allclose(result, expected)
