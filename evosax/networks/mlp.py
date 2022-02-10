import jax
import jax.numpy as jnp
from flax import linen as nn


# MLP - Works on batches of data! Supervised Learning Tasks
# TanhMLP, ContinuousMLP, DiscreteMLP - Work on raw observations (RL)


class MLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int
    model_name: str = "MLP"

    @nn.compact
    def __call__(self, x, rng):
        x = x.reshape((x.shape[0], -1))
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_output_units)(x)
        return x


class TanhMLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int
    model_name: str = "TanhMLP"

    @nn.compact
    def __call__(self, x, rng):
        x = x.ravel()
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_output_units)(x)
        return nn.tanh(x)


class ContinuousMLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int
    model_name: str = "ContinuousMLP"

    @nn.compact
    def __call__(self, x, rng):
        x = x.ravel()
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x_mean = nn.Dense(features=self.num_output_units)(x)
        x_log_var = nn.Dense(features=1)(x)
        x_std = jnp.exp(0.5 * x_log_var)
        noise = x_std * jax.random.normal(rng, (self.num_output_units,))
        x_out = x_mean + noise
        return x_out


class DiscreteMLP(nn.Module):
    """Simple ReLU MLP."""

    num_hidden_units: int
    num_hidden_layers: int
    num_output_units: int
    model_name: str = "DiscreteMLP"

    @nn.compact
    def __call__(self, x, rng):
        x = x.ravel()
        for l in range(self.num_hidden_layers):
            x = nn.Dense(features=self.num_hidden_units)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.num_output_units)(x)
        x_out = jax.random.categorical(rng, x)
        return x_out
