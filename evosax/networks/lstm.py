from flax import linen as nn
import jax
import jax.numpy as jnp


class LSTM(nn.Module):
    num_hidden_units: int
    num_output_units: int
    model_name: str = "LSTM"

    @nn.compact
    def __call__(self, x, carry, rng):
        lstm_state, rng = carry
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        x = nn.Dense(features=self.num_output_units)(y)
        return lstm_state, x

    def initialize_carry(self):
        # Use fixed random key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.num_hidden_units
        )


class TanhLSTM(nn.Module):
    num_hidden_units: int
    num_output_units: int
    model_name: str = "TanhLSTM"

    @nn.compact
    def __call__(self, x, carry, rng):
        lstm_state = carry
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        x = nn.Dense(features=self.num_output_units)(y)
        return lstm_state, nn.tanh(x)

    def initialize_carry(self):
        # Use fixed random key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.num_hidden_units
        )


class ContinuousLSTM(nn.Module):
    num_hidden_units: int
    num_output_units: int
    model_name: str = "ContinuousLSTM"

    @nn.compact
    def __call__(self, x, carry, rng):
        lstm_state = carry
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        x_mean = nn.Dense(features=self.num_output_units)(y)
        x_log_var = nn.Dense(features=1)(y)
        x_std = jnp.exp(0.5 * x_log_var)
        noise = x_std * jax.random.normal(rng, (self.num_output_units,))
        x_out = x_mean + noise
        return lstm_state, x_out

    def initialize_carry(self):
        # Use fixed random key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.num_hidden_units
        )


class DiscreteLSTM(nn.Module):
    num_hidden_units: int
    num_output_units: int
    model_name: str = "DiscreteLSTM"

    @nn.compact
    def __call__(self, x, carry, rng):
        lstm_state = carry
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        x = nn.Dense(features=self.num_output_units)(y)
        x_out = jax.random.categorical(rng, x)
        return lstm_state, x_out

    def initialize_carry(self):
        # Use fixed random key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.num_hidden_units
        )
