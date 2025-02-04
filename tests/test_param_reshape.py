import jax
import jax.numpy as jnp
from evosax import ParameterReshaper
from evosax.networks import CNN, MLP


def test_flat_vector():
    rng = jax.random.PRNGKey(0)
    vec_params = jax.random.normal(rng, (2,))
    reshaper = ParameterReshaper(vec_params)
    assert reshaper.total_params == 2

    # Test population batch matrix reshaping
    test_params = jnp.zeros((100, 2))
    out = reshaper.reshape(test_params)
    assert out.shape == (100, 2)


# def test_reshape_lstm():
#     rng = jax.random.PRNGKey(1)
#     network = LSTM(
#         num_hidden_units=10,
#         num_output_units=1,
#         output_activation="identity",
#     )
#     pholder = jnp.zeros((10,))
#     carry_init = network.initialize_carry()
#     net_params = network.init(
#         rng,
#         x=pholder,
#         carry=carry_init,
#         rng=rng,
#     )

#     reshaper = ParameterReshaper(net_params)
#     # net_params["params"]["LSTMCell_0"].keys()
#     assert reshaper.total_params == 851

#     # Test population batch matrix reshaping
#     test_params = jnp.zeros((100, 851))
#     out = reshaper.reshape(test_params)
#     assert out["params"]["LSTMCell_0"]["hf"]["kernel"].shape == (100, 10, 10)

#     # Test single network vector reshaping
#     test_single = jnp.zeros(851)
#     out = reshaper.reshape_single(test_single)
#     assert out["params"]["LSTMCell_0"]["hf"]["kernel"].shape == (10, 10)


def test_reshape_mlp():
    rng = jax.random.PRNGKey(1)
    network = MLP(
        num_hidden_units=64,
        num_hidden_layers=2,
        num_output_units=1,
        hidden_activation="relu",
        output_activation="identity",
    )
    pholder = jnp.zeros((10,))
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )

    reshaper = ParameterReshaper(net_params)
    assert reshaper.total_params == (10 * 64 + 64 + 64 * 64 + 64 + 64 + 1)

    # Test population batch matrix reshaping
    test_params = jnp.zeros((100, 4929))
    out = reshaper.reshape(test_params)
    assert out["params"]["Dense_0"]["kernel"].shape == (100, 10, 64)

    # Test single network vector reshaping
    test_single = jnp.zeros(4929)
    out = reshaper.reshape_single(test_single)
    assert out["params"]["Dense_0"]["kernel"].shape == (10, 64)


def test_reshape_cnn():
    rng = jax.random.PRNGKey(1)
    network = CNN(
        depth_1=1,
        depth_2=1,
        features_1=16,
        features_2=8,
        kernel_1=3,
        kernel_2=5,
        strides_1=1,
        strides_2=1,
        num_linear_layers=1,
        num_hidden_units=16,
        num_output_units=10,
    )
    pholder = jnp.zeros((1, 28, 28, 1))
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )

    reshaper = ParameterReshaper(net_params)
    assert reshaper.total_params == 9826

    # Test population batch matrix reshaping
    test_params = jnp.zeros((100, 9826))
    out = reshaper.reshape(test_params)
    assert out["params"]["Conv_0"]["kernel"].shape == (100, 3, 3, 1, 16)

    # Test single network vector reshaping
    test_single = jnp.zeros(9826)
    out = reshaper.reshape_single(test_single)
    assert out["params"]["Conv_0"]["kernel"].shape == (3, 3, 1, 16)
