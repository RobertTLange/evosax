import jax
from evosax import CMA_ES
from evosax.problems import ClassicFitness, GymnaxFitness, BraxFitness


def test_classic_rollout(classic_name: str):
    # Instantiate the search strategy
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


def test_gym_rollout(gym_name: str):
    # TODO: Implement gym rollout test - GymnaxFitness
    return


def test_brax_rollout(brax_name: str):
    # TODO: Implement brax rollout test - BraxFitness
    return
