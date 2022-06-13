import jax
import jax.numpy as jnp
from evosax import CMA_ES, ARS, ParameterReshaper, NetworkMapper
from evosax.problems import (
    ClassicFitness,
    GymFitness,
    BraxFitness,
    VisionFitness,
    SequenceFitness,
)


def test_classic_rollout(classic_name: str):
    rng = jax.random.PRNGKey(0)
    evaluator = ClassicFitness(
        classic_name, num_dims=2, num_rollouts=2, noise_std=0.1
    )
    strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, params)
    fitness = evaluator.rollout(rng_eval, x)
    assert fitness.shape == (20, 2)


def test_env_ffw_rollout(env_name: str):
    rng = jax.random.PRNGKey(0)
    if env_name in ["CartPole-v1"]:
        evaluator = GymFitness(env_name, num_env_steps=100, num_rollouts=10)
        network = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            num_output_units=evaluator.action_shape,
            hidden_activation="relu",
            output_activation="categorical",
        )
    else:
        evaluator = BraxFitness(env_name, num_env_steps=100, num_rollouts=10)
        network = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            num_output_units=evaluator.action_shape,
            hidden_activation="tanh",
            output_activation="tanh",
        )
    pholder = jnp.zeros((1, evaluator.input_shape[0]))
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )
    reshaper = ParameterReshaper(net_params)
    evaluator.set_apply_fn(reshaper.vmap_dict, network.apply)

    strategy = ARS(popsize=20, num_dims=reshaper.total_params, elite_ratio=0.5)
    state = strategy.initialize(rng)
    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state)
    x_re = reshaper.reshape(x)
    fitness = evaluator.rollout(rng_eval, x_re)

    # Assert shape (#popmembers, #rollouts)
    assert fitness.shape == (20, 10)


def test_env_rec_rollout(env_name: str):
    rng = jax.random.PRNGKey(0)
    if env_name in ["CartPole-v1"]:
        evaluator = GymFitness(env_name, num_env_steps=100, num_rollouts=10)
        network = NetworkMapper["LSTM"](
            num_hidden_units=64,
            num_output_units=evaluator.action_shape,
            output_activation="categorical",
        )

    else:
        evaluator = BraxFitness(env_name, num_env_steps=100, num_rollouts=10)
        network = NetworkMapper["LSTM"](
            num_hidden_units=64,
            num_output_units=evaluator.action_shape,
            output_activation="tanh",
        )

    pholder = jnp.zeros((1, evaluator.input_shape[0]))
    carry_init = network.initialize_carry()
    net_params = network.init(
        rng,
        x=pholder,
        carry=carry_init,
        rng=rng,
    )
    reshaper = ParameterReshaper(net_params)
    evaluator.set_apply_fn(
        reshaper.vmap_dict, network.apply, network.initialize_carry
    )
    strategy = ARS(popsize=20, num_dims=reshaper.total_params, elite_ratio=0.5)
    state = strategy.initialize(rng)

    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state)
    x_re = reshaper.reshape(x)
    fitness = evaluator.rollout(rng_eval, x_re)

    # Assert shape (#popmembers, #rollouts)
    assert fitness.shape == (20, 10)


def test_vision_fitness():
    rng = jax.random.PRNGKey(0)
    evaluator = VisionFitness("MNIST", 4, test=True)
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
    pholder = jnp.zeros((1, 28, 28, 1))
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )

    reshaper = ParameterReshaper(net_params)
    evaluator.set_apply_fn(reshaper.vmap_dict, network.apply)

    strategy = ARS(popsize=4, num_dims=reshaper.total_params, elite_ratio=0.5)
    state = strategy.initialize(rng)

    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state)
    x_re = reshaper.reshape(x)
    loss, acc = evaluator.rollout(rng_eval, x_re)
    assert loss.shape == (4, 1)
    assert acc.shape == (4, 1)


def test_sequence_fitness():
    rng = jax.random.PRNGKey(0)
    evaluator = SequenceFitness(task_name="Addition", batch_size=10, test=False)
    network = NetworkMapper["LSTM"](
        num_hidden_units=100,
        num_output_units=evaluator.action_shape,
    )
    params = network.init(
        rng,
        x=jnp.ones([1, evaluator.input_shape[0]]),
        carry=network.initialize_carry(),
        rng=rng,
    )
    param_reshaper = ParameterReshaper(params)
    evaluator.set_apply_fn(
        param_reshaper.vmap_dict,
        network.apply,
        network.initialize_carry,
    )

    strategy = ARS(param_reshaper.total_params, 4)
    (param_reshaper.total_params)
    es_state = strategy.initialize(rng)

    x, es_state = strategy.ask(rng, es_state)
    reshaped_params = param_reshaper.reshape(x)
    # Rollout population performance, reshape fitness & update strategy.
    loss, perf = evaluator.rollout(rng, reshaped_params)
    assert loss.shape == (4, 1)
    assert perf.shape == (4, 1)
