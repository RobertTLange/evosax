import jax
import jax.numpy as jnp
from flax import linen as nn
from evosax.utils import ParameterReshaper


class LSTM(nn.Module):
    num_hidden_units: int
    num_output_units: int
    model_name: str = "LSTM"

    @nn.compact
    def __call__(self, carry, x, rng):
        lstm_state = carry
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        logits = nn.Dense(features=self.num_output_units)(y)
        return lstm_state, logits

    def initialize_carry(self):
        # Use fixed random key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), self.num_hidden_units
        )


def test_reshape_lstm():
    rng = jax.random.PRNGKey(1)
    network = LSTM(num_hidden_units=10, num_output_units=1)
    hidden = network.initialize_carry()
    net_params = network.init(
        rng,
        hidden,
        jnp.zeros(
            10,
        ),
        rng=rng,
    )

    reshaper = ParameterReshaper(net_params["params"])
    # net_params["params"]["LSTMCell_0"].keys()
    assert reshaper.total_params == 851

    # Test population batch matrix reshaping
    test_params = jnp.zeros((100, 531))
    out = reshaper.reshape(test_params)
    assert out["LSTMCell_0"]["hf"]["kernel"].shape == (100, 10, 10)

    # Test single network vector reshaping
    test_single = jnp.zeros(531)
    out = reshaper.reshape_single(test_single)
    assert out["LSTMCell_0"]["hf"]["kernel"].shape == (10, 10)
    return
