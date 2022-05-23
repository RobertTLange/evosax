import jax
import jax.numpy as jnp
from evosax.networks import LSTM
from evosax.experimental.decodings import RandomDecoder


def test_random_decoding():
    rng = jax.random.PRNGKey(0)
    num_encoding_dims = 10
    popsize = 100
    # Define some network architecture to optimize params for
    network = LSTM(
        num_hidden_units=10,
        num_output_units=1,
        output_activation="identity",
    )
    net_params = network.init(
        rng,
        x=jnp.zeros((10,)),
        carry=network.initialize_carry(),
        rng=rng,
    )

    # Define candidates coming from strategy.ask - batch reshape
    encodings = jnp.zeros((popsize, num_encoding_dims))
    decoder = RandomDecoder(num_encoding_dims, net_params["params"])
    x = decoder.reshape(encodings)
    assert x["LSTMCell_0"]["hf"]["kernel"].shape == (popsize, 10, 10)

    # Define candidates coming from strategy.ask - single reshape
    encodings = jnp.zeros(num_encoding_dims)
    decoder = RandomDecoder(num_encoding_dims, net_params["params"])
    x = decoder.reshape_single(encodings)
    assert x["LSTMCell_0"]["hf"]["kernel"].shape == (10, 10)
