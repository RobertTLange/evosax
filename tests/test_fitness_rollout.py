import jax
import jax.numpy as jnp
from evosax import CMA_ES, Augmented_RS, ParameterReshaper, NetworkMapper
from evosax.problems import ClassicFitness, GymFitness, BraxFitness


def test_classic_rollout(classic_name: str):
    rng = jax.random.PRNGKey(0)
    evaluator = ClassicFitness(classic_name, num_dims=2)
    strategy = CMA_ES(popsize=20, num_dims=2, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.initialize(rng, params)

    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, state = strategy.ask(rng_gen, state, params)
    fitness = evaluator.rollout(rng_eval, x)
    assert fitness.shape == (20,)


def test_gymnax_ffw_rollout(gymnax_name: str):
    rng = jax.random.PRNGKey(0)
    evaluator = GymFitness()
    network = NetworkMapper["MLP"](
        num_hidden_units=64,
        num_hidden_layers=2,
        num_output_units=2,
        hidden_activation="relu",
        output_activation="categorical",
    )
    pholder = jnp.zeros(evaluator.input_shape)
    net_params = network.init(
        rng,
        x=pholder,
        rng=rng,
    )
    evaluator.set_apply_fn(network.apply)
    reshaper = ParameterReshaper(net_params["params"])
    strategy = Augmented_RS(
        popsize=20, num_dims=reshaper.total_params, elite_ratio=0.5
    )
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    rollout = jax.vmap(evaluator.rollout, in_axes=(None, reshaper.vmap_dict))
    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    rng_epi = jax.random.split(rng_eval, 5)
    x, state = strategy.ask(rng_gen, state, params)
    x_re = reshaper.reshape(x)
    fitness = rollout(rng_epi, x_re)
    assert fitness.shape == (20, 5)


def test_gymnax_rec_rollout(gymnax_name: str):
    rng = jax.random.PRNGKey(0)
    evaluator = GymFitness()
    network = NetworkMapper["LSTM"](
        num_hidden_units=64,
        num_output_units=2,
        output_activation="categorical",
    )
    pholder = jnp.zeros(evaluator.input_shape)
    carry_init = network.initialize_carry()
    net_params = network.init(
        rng,
        x=pholder,
        carry=carry_init,
        rng=rng,
    )
    evaluator.set_apply_fn(network.apply, network.initialize_carry)
    reshaper = ParameterReshaper(net_params["params"])
    strategy = Augmented_RS(
        popsize=20, num_dims=reshaper.total_params, elite_ratio=0.5
    )
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    rollout = jax.vmap(evaluator.rollout, in_axes=(None, reshaper.vmap_dict))
    # Run the ask-eval-tell loop
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    rng_epi = jax.random.split(rng_eval, 5)
    x, state = strategy.ask(rng_gen, state, params)
    x_re = reshaper.reshape(x)
    fitness = rollout(rng_epi, x_re)
    assert fitness.shape == (20, 5)
