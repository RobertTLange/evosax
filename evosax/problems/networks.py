"""Neural network architectures for problems."""

from collections.abc import Callable, Sequence

import jax
from flax import linen as nn


def identity_output_fn(x: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """Identity output function."""
    return x


def categorical_output_fn(x: jax.Array, key: jax.Array) -> jax.Array:
    """Categorical sample from logits."""
    return jax.random.categorical(key, x)


def tanh_output_fn(x: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """Tanh output function."""
    return nn.tanh(x)


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    kernel_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Callable = nn.initializers.zeros_init()
    activation: Callable = nn.tanh
    output_fn: Callable = categorical_output_fn
    use_bias: bool = True
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, key: jax.Array | None = None):
        """Forward pass of the MLP."""
        hidden = x
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = nn.Dense(
                hidden_size,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(hidden)

            if i < len(self.layer_sizes) - 1:
                hidden = self.activation(hidden)
            else:
                hidden = self.output_fn(hidden, key)

        if self.layer_norm:
            hidden = nn.LayerNorm()(hidden)

        return hidden


class CNN(nn.Module):
    """CNN module."""

    num_filters: Sequence[int]
    kernel_sizes: Sequence[tuple]
    strides: Sequence[tuple]
    mlp_layer_sizes: Sequence[int]
    kernel_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Callable = nn.initializers.zeros_init()
    activation: Callable = nn.relu
    output_fn: Callable = categorical_output_fn
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array, key: jax.Array | None = None):
        """Forward pass of the CNN."""
        hidden = x
        for num_filter, kernel_size, stride in zip(
            self.num_filters, self.kernel_sizes, self.strides
        ):
            hidden = nn.Conv(
                num_filter,
                kernel_size=kernel_size,
                strides=stride,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(hidden)
            hidden = self.activation(hidden)

        hidden = hidden.reshape(x.shape[0], -1)
        return MLP(
            layer_sizes=self.mlp_layer_sizes,
            activation=self.activation,
            output_fn=self.output_fn,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(hidden, key)
