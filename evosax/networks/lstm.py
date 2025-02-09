import jax
from flax import linen as nn

from ..types import ArrayTree
from .shared import (
    categorical_out,
    default_bias_init,
    gaussian_out,
    identity_out,
    kernel_init_fn,
    tanh_out,
)


class LSTM(nn.Module):
    """Simple LSTM Wrapper with flexible output head."""

    num_hidden_units: int = 32
    num_output_units: int = 1
    output_activation: str = "identity"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "LSTM"

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        carry: ArrayTree,
        key: jax.Array | None = None,
    ) -> tuple[tuple[ArrayTree, ArrayTree], jax.Array]:
        lstm_state, x = nn.LSTMCell(
            bias_init=default_bias_init(),
            kernel_init=kernel_init_fn[self.kernel_init_type](),
        )(carry, x)
        if self.output_activation == "identity":
            x = identity_out(x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "tanh":
            x = tanh_out(x, self.num_output_units, self.kernel_init_type)
        # Categorical and gaussian output heads require random key for sampling
        elif self.output_activation == "categorical":
            x = categorical_out(key, x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "gaussian":
            x = gaussian_out(key, x, self.num_output_units, self.kernel_init_type)
        return lstm_state, x

    def init_carry(self) -> tuple[ArrayTree, ArrayTree]:
        """Initialize hidden state of LSTM."""
        return nn.LSTMCell.initialize_carry(
            jax.random.key(0), (), self.num_hidden_units
        )
