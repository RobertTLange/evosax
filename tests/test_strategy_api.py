import jax
from evosax import Strategies
from evosax.problems import BBOBFitness


def test_strategy_ask(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    if strategy_name == "ESMC":
        popsize = 21
    else:
        popsize = 20
    strategy = Strategies[strategy_name](popsize=popsize, num_dims=2)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    x, state = strategy.ask(rng, state, params)
    assert x.shape[0] == popsize
    assert x.shape[1] == 2
    return


def test_strategy_ask_tell(strategy_name):
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    if strategy_name == "ESMC":
        popsize = 21
    else:
        popsize = 20
    strategy = Strategies[strategy_name](popsize=popsize, num_dims=2)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    x, state = strategy.ask(rng, state, params)
    evaluator = BBOBFitness("Sphere", num_dims=2)
    fitness = evaluator.rollout(rng, x)
    state = strategy.tell(x, fitness, state, params)
    return
