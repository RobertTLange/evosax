import jax.numpy as jnp
from flax import linen as nn
import chex
from typing import Tuple
from .shared import default_bias_init, kernel_init_fn


def conv_relu_block(
    x: chex.Array,
    features: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str = "SAME",
    kernel_init_type: str = "lecun_normal",
) -> chex.Array:
    """Convolution layer + ReLU activation."""
    x = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=True,
        padding=padding,
        bias_init=default_bias_init(),
        kernel_init=kernel_init_fn[kernel_init_type](),
    )(x)
    x = nn.relu(x)
    return x


def conv_relu_pool_block(
    x: chex.Array,
    features: int,
    kernel_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str = "SAME",
    kernel_init_type: str = "lecun_normal",
) -> chex.Array:
    """Convolution layer + ReLU activation + Avg. Pooling."""
    x = nn.Conv(
        features=features,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=True,
        padding=padding,
        bias_init=default_bias_init(),
        kernel_init=kernel_init_fn[kernel_init_type](),
    )(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    return x


class CNN(nn.Module):
    """Simple CNN Wrapper with Conv-ReLu-MaxPool blocks."""

    num_output_units: int = 10
    depth_1: int = 1
    depth_2: int = 1
    features_1: int = 8
    features_2: int = 16
    kernel_1: int = 5
    kernel_2: int = 5
    strides_1: int = 1
    strides_2: int = 1
    num_linear_layers: int = 1
    num_hidden_units: int = 16
    kernel_init_type: str = "lecun_normal"
    model_name: str = "CNN"

    @nn.compact
    def __call__(self, x: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        # Block In 1:
        for i in range(self.depth_1):
            x = conv_relu_pool_block(
                x,
                self.features_1,
                (self.kernel_1, self.kernel_1),
                (self.strides_1, self.strides_1),
                kernel_init_type=self.kernel_init_type,
            )

        # Block In 2:
        for i in range(self.depth_2):
            x = conv_relu_pool_block(
                x,
                self.features_2,
                (self.kernel_2, self.kernel_2),
                (self.strides_2, self.strides_2),
                kernel_init_type=self.kernel_init_type,
            )
        x = x.reshape((x.shape[0], -1))
        # Squeeze and linear layers
        for l in range(self.num_linear_layers):
            x = nn.Dense(
                features=self.num_hidden_units,
                bias_init=default_bias_init(),
                kernel_init=kernel_init_fn[self.kernel_init_type](),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            features=self.num_output_units,
            bias_init=default_bias_init(),
            kernel_init=kernel_init_fn[self.kernel_init_type](),
        )(x)
        return x


class All_CNN_C(nn.Module):
    """All-CNN-inspired architecture as in Springenberg et al. (2015).
    Reference: https://arxiv.org/abs/1412.6806"""

    num_output_units: int = 10
    depth_1: int = 1
    depth_2: int = 1
    features_1: int = 8
    features_2: int = 16
    kernel_1: int = 5
    kernel_2: int = 5
    strides_1: int = 1
    strides_2: int = 1
    final_window: Tuple[int, int] = (28, 28)
    kernel_init_type: str = "lecun_normal"
    model_name: str = "All_CNN_C"

    @nn.compact
    def __call__(self, x: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        # Block In 1:
        for i in range(self.depth_1):
            x = conv_relu_block(
                x,
                self.features_1,
                (self.kernel_1, self.kernel_1),
                (self.strides_1, self.strides_1),
                kernel_init_type=self.kernel_init_type,
            )

        # Block In 2:
        for i in range(self.depth_2):
            x = conv_relu_block(
                x,
                self.features_2,
                (self.kernel_2, self.kernel_2),
                (self.strides_2, self.strides_2),
                kernel_init_type=self.kernel_init_type,
            )

        # Block Out: 1 × 1 conv. num_outputs-ReLu ×n
        x = nn.Conv(
            features=self.num_output_units,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=True,
            padding="SAME",
            bias_init=default_bias_init(),
            kernel_init=kernel_init_fn[self.kernel_init_type](),
        )(x)

        # Global average pooling -> logits
        x = nn.avg_pool(
            x, window_shape=self.final_window, strides=None, padding="VALID"
        )
        x = jnp.squeeze(x, axis=(1, 2))
        return x
