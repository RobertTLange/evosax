from flax import linen as nn
import chex
from typing import Optional
from .shared import (
    identity_out,
    tanh_out,
    categorical_out,
    gaussian_out,
    default_bias_init,
    kernel_init_fn,
)


class MLP(nn.Module):
    """Simple MLP Wrapper with flexible output head."""

    num_hidden_units: int = 64
    num_hidden_layers: int = 2
    num_output_units: int = 1
    hidden_activation: str = "relu"
    output_activation: str = "identity"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "MLP"

    @nn.compact
    def __call__(
        self, x: chex.Array, rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        for l in range(self.num_hidden_layers):
            x = nn.Dense(
                features=self.num_hidden_units,
                kernel_init=kernel_init_fn[self.kernel_init_type](),
                bias_init=default_bias_init(),
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
            return identity_out(x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "tanh":
            return tanh_out(x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "categorical":
            return categorical_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        elif self.output_activation == "gaussian":
            return gaussian_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
