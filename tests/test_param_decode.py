import jax
import jax.numpy as jnp
from evosax.networks import LSTM, MLP
from evosax.experimental.decodings import RandomDecoder, HyperDecoder


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
    decoder = RandomDecoder(num_encoding_dims, net_params)
    x = decoder.reshape(encodings)
    assert x["params"]["LSTMCell_0"]["hf"]["kernel"].shape == (popsize, 10, 10)

    # Define candidates coming from strategy.ask - single reshape
    encodings = jnp.zeros(num_encoding_dims)
    decoder = RandomDecoder(num_encoding_dims, net_params)
    x = decoder.reshape_single(encodings)
    assert x["params"]["LSTMCell_0"]["hf"]["kernel"].shape == (10, 10)


def test_hypermlp_decoding():
    rng = jax.random.PRNGKey(0)
    popsize = 100
    # Define some network architecture to optimize params for
    network = MLP(
        num_hidden_units=64,
        num_hidden_layers=2,
        num_output_units=1,
        output_activation="identity",
    )
    net_params = network.init(
        rng,
        x=jnp.zeros((10,)),
        rng=rng,
    )

    # Define candidates coming from strategy.ask - batch reshape
    decoder = HyperDecoder(
        net_params,
        hypernet_config={
            "num_latent_units": 3,  # Latent units per module kernel/bias
            "num_hidden_units": 2,  # Hidden dimensionality of a_i^j embedding
        },
    )
    encodings = jnp.zeros((popsize, decoder.num_encoding_dims))
    x = decoder.reshape(encodings)
    assert x["params"]["Dense_0"]["kernel"].shape == (popsize, 10, 64)

    # Define candidates coming from strategy.ask - single reshape
    encodings = jnp.zeros(decoder.num_encoding_dims)
    x = decoder.reshape_single(encodings)
    assert x["params"]["Dense_0"]["kernel"].shape == (10, 64)
