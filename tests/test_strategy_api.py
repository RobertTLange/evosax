import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.problems import BBOBFitness


def test_strategy_ask(strategy_name):
    num_dims = 2
    x = jnp.zeros((num_dims,))

    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    popsize = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        strategy = Strategies[strategy_name](
            npop=1, subpopsize=popsize, pholder_params=x
        )
    else:
        strategy = Strategies[strategy_name](popsize=popsize, pholder_params=x)
    params = strategy.default_params
    state = strategy.initialize(key, params)
    x, state = strategy.ask(key, state, params)
    assert x.shape[0] == popsize
    assert x.shape[1] == 2
    return


def test_strategy_ask_tell(strategy_name):
    num_dims = 2
    x = jnp.zeros((num_dims,))

    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    popsize = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        strategy = Strategies[strategy_name](
            npop=1, subpopsize=popsize, pholder_params=x
        )
    else:
        strategy = Strategies[strategy_name](popsize=popsize, pholder_params=x)
    params = strategy.default_params
    state = strategy.initialize(key, params)
    x, state = strategy.ask(key, state, params)
    evaluator = BBOBFitness("sphere", num_dims)
    fitness = evaluator.rollout(key, x)
    state = strategy.tell(x, fitness, state, params)
    return
