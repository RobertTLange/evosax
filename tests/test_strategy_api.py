import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.problems import BBOBProblem


def test_strategy_ask(strategy_name):
    num_dims = 2
    x = jnp.zeros((num_dims,))

    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    population_size = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        strategy = Strategies[strategy_name](
            npop=1, subpopulation_size=population_size, solution=x
        )
    else:
        strategy = Strategies[strategy_name](
            population_size=population_size, solution=x
        )
    params = strategy.default_params
    state = strategy.init(key, params)
    x, state = strategy.ask(key, state, params)
    assert x.shape[0] == population_size
    assert x.shape[1] == 2
    return


def test_strategy_ask_tell(strategy_name):
    num_dims = 2
    x = jnp.zeros((num_dims,))

    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    population_size = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        strategy = Strategies[strategy_name](
            npop=1, subpopulation_size=population_size, solution=x
        )
    else:
        strategy = Strategies[strategy_name](
            population_size=population_size, solution=x
        )
    params = strategy.default_params
    state = strategy.init(key, params)
    x, state = strategy.ask(key, state, params)
    problem = BBOBProblem("sphere", num_dims)
    fitness = problem.eval(key, x)
    state = strategy.tell(x, fitness, state, params)
    return
