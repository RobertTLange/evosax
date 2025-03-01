import jax
import jax.numpy as jnp
from evosax import ARS, CMA_ES, NetworkMapper
from evosax.problems import (
    BBOBProblem,
    GymnaxProblem,
    VisionProblem,
)


def test_bbob_eval(classic_name: str):
    key = jax.random.key(0)
    problem = BBOBProblem(classic_name, num_dims=2)
    x = problem.sample(key)
    strategy = CMA_ES(population_size=20, solution=x, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.init(key, params)

    # Run the ask-eval-tell loop
    key_ask, key_eval = jax.random.split(key)
    x, state = strategy.ask(key_ask, state, params)
    fitness = problem.eval(key_eval, x)
    assert fitness.shape == (20,)


def test_env_ffw_eval(env_name: str):
    key = jax.random.key(0)
    problem = GymnaxProblem(env_name, episode_length=100, num_rollouts=10)
    network = NetworkMapper["MLP"](
        num_hidden_units=64,
        num_hidden_layers=2,
        num_output_units=problem.action_shape,
        hidden_activation="relu",
        output_activation="categorical",
    )
    solution = jnp.zeros((1, problem.input_shape[0]))
    net_params = network.init(
        key,
        x=solution,
        key=key,
    )
    problem.set_apply_fn(network.apply)

    strategy = ARS(population_size=20, solution=net_params, elite_ratio=0.5)
    state = strategy.init(key)
    # Run the ask-eval-tell loop
    key_ask, key_eval = jax.random.split(key)
    x, state = strategy.ask(key_ask, state)
    fitness = problem.eval(key_eval, x)

    # Assert shape (#popmembers, #rollouts)
    assert fitness.shape == (20, 10)


def test_vision_fitness():
    key = jax.random.key(0)
    problem = VisionProblem("MNIST", 4, test=True)
    network = NetworkMapper["CNN"](
        depth_1=1,
        depth_2=1,
        features_1=8,
        features_2=16,
        kernel_1=5,
        kernel_2=5,
        strides_1=1,
        strides_2=1,
        num_linear_layers=0,
        num_output_units=10,
    )
    # Channel last configuration for conv!
    solution = jnp.zeros((1, 28, 28, 1))
    net_params = network.init(
        key,
        x=solution,
        key=key,
    )

    problem.set_apply_fn(network.apply)

    strategy = ARS(population_size=4, solution=net_params, elite_ratio=0.5)
    state = strategy.init(key)

    # Run the ask-eval-tell loop
    key_ask, key_eval = jax.random.split(key)
    x, state = strategy.ask(key_ask, state)
    loss, acc = problem.eval(key_eval, x)
    assert loss.shape == (4, 1)
    assert acc.shape == (4, 1)
