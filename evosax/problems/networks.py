"""Neural network architectures for problems."""

from collections.abc import Callable, Sequence

import jax
from flax import linen as nn


def identity_output_fn(x: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """Identity output function.

    Args:
        x: Output of the last network layer.
        key: Unused. Present so that all output functions share one signature.

    Returns:
        `x` unchanged.
    """
    return x


def categorical_output_fn(x: jax.Array, key: jax.Array) -> jax.Array:
    """Categorical sample from logits.

    Args:
        x: Unnormalized logits. Categories are indexed along the last axis.
        key: PRNG key.

    Returns:
        The sampled category indices.
    """
    return jax.random.categorical(key, x)


def tanh_output_fn(x: jax.Array, key: jax.Array | None = None) -> jax.Array:
    """Tanh output function.

    Args:
        x: Output of the last network layer.
        key: Unused. Present so that all output functions share one signature.

    Returns:
        `tanh(x)`, computed elementwise.
    """
    return nn.tanh(x)


class MLP(nn.Module):
    """MLP module.

    A stack of dense layers. `activation` follows every layer except the last,
    whose output is passed through `output_fn`.

    Attributes:
        layer_sizes: Number of units of every dense layer. The last entry is the
            output size.
        kernel_init: Initializer of the dense kernels.
        bias_init: Initializer of the dense biases.
        activation: Activation function applied after every layer except the
            last.
        output_fn: Function `(x, key) -> output` applied to the output of the
            last layer, e.g. `identity_output_fn`, `categorical_output_fn` or
            `tanh_output_fn`.
        use_bias: If True, the dense layers have a bias.
        layer_norm: If True, apply layer normalization to the result of
            `output_fn`.
    """

    layer_sizes: Sequence[int]
    kernel_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Callable = nn.initializers.zeros_init()
    activation: Callable = nn.tanh
    output_fn: Callable = categorical_output_fn
    use_bias: bool = True
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, key: jax.Array | None = None):
        """Forward pass of the MLP.

        Args:
            x: Input array; the dense layers act on its last axis.
            key: PRNG key passed to `output_fn`.

        Returns:
            The result of `output_fn`.
        """
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
    """CNN module.

    A stack of convolutional layers, each followed by `activation`, whose
    flattened output is passed to an `MLP` head.

    Attributes:
        num_filters: Number of filters of every convolutional layer.
        kernel_sizes: Kernel size of every convolutional layer.
        strides: Strides of every convolutional layer.
        mlp_layer_sizes: Number of units of every dense layer of the MLP head.
            The last entry is the output size.
        kernel_init: Initializer of the convolutional and dense kernels.
        bias_init: Initializer of the convolutional and dense biases.
        activation: Activation function applied after every convolutional layer
            and inside the MLP head.
        output_fn: Function `(x, key) -> output` applied to the output of the
            last dense layer, e.g. `identity_output_fn`, `categorical_output_fn`
            or `tanh_output_fn`.
        use_bias: If True, the convolutional and dense layers have a bias.
    """

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
        """Forward pass of the CNN.

        Args:
            x: Batch of inputs with a leading batch axis and a trailing channel
                axis.
            key: PRNG key passed to `output_fn`.

        Returns:
            The result of `output_fn`, with one row per batch element.
        """
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
