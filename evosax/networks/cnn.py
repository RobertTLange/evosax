import jax
import jax.numpy as jnp
from flax import linen as nn
import chex
from typing import Tuple, Optional
from .shared import (
    identity_out,
    tanh_out,
    categorical_out,
    gaussian_out,
    default_bias_init,
    kernel_init_fn,
)


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
    """Simple CNN Wrapper with Conv-ReLu-AvgPool blocks."""

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
    hidden_activation: str = "relu"
    output_activation: str = "identity"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "CNN"

    @nn.compact
    def __call__(
        self, x: chex.Array, rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        # Add batch dimension if only processing single 3d array
        if len(x.shape) < 4:
            x = jnp.expand_dims(x, 0)
            batch_case = False
        else:
            batch_case = True

        # Block In 1:
        for _ in range(self.depth_1):
            x = conv_relu_pool_block(
                x,
                self.features_1,
                (self.kernel_1, self.kernel_1),
                (self.strides_1, self.strides_1),
                kernel_init_type=self.kernel_init_type,
            )

        # Block In 2:
        for _ in range(self.depth_2):
            x = conv_relu_pool_block(
                x,
                self.features_2,
                (self.kernel_2, self.kernel_2),
                (self.strides_2, self.strides_2),
                kernel_init_type=self.kernel_init_type,
            )
        # Flatten the output into vector for Dense Readout
        x = x.reshape(x.shape[0], -1)
        # Squeeze and linear layers
        for _ in range(self.num_linear_layers):
            x = nn.Dense(
                features=self.num_hidden_units,
                bias_init=default_bias_init(),
                kernel_init=kernel_init_fn[self.kernel_init_type](),
            )(x)
            if self.hidden_activation == "relu":
                x = nn.relu(x)
            elif self.hidden_activation == "tanh":
                x = nn.tanh(x)
            elif self.hidden_activation == "gelu":
                x = nn.gelu(x)
            elif self.hidden_activation == "softplus":
                x = nn.softplus(x)

        if self.output_activation == "identity":
            x = identity_out(x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "tanh":
            x = tanh_out(x, self.num_output_units, self.kernel_init_type)
        # Categorical and gaussian output heads require rng for sampling
        elif self.output_activation == "categorical":
            x = categorical_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        elif self.output_activation == "gaussian":
            x = gaussian_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        # Squeeze away extra dimension - e.g. single action output for RL
        if not batch_case:
            return x.squeeze()
        else:
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
    output_activation: str = "identity"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "All_CNN_C"

    @nn.compact
    def __call__(
        self, x: chex.Array, rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        # Add batch dimension if only processing single 3d array
        if len(x.shape) < 4:
            x = jnp.expand_dims(x, 0)
            batch_case = False
        else:
            batch_case = True
        # Block In 1:
        for _ in range(self.depth_1):
            x = conv_relu_block(
                x,
                self.features_1,
                (self.kernel_1, self.kernel_1),
                (self.strides_1, self.strides_1),
                kernel_init_type=self.kernel_init_type,
            )

        # Block In 2:
        for _ in range(self.depth_2):
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

        if self.output_activation == "tanh":
            x = nn.tanh(x)
        # Categorical head requires rng for sampling
        elif self.output_activation == "categorical":
            x = jax.random.categorical(rng, x)
        # No gaussian option implemented so far - need second 1x1 conv + pool
        # Squeeze away extra dimension - e.g. single action output for RL
        if not batch_case:
            return x.squeeze()
        else:
            return x
