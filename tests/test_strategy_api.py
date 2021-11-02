import jax
from evosax import Strategies
from evosax.problems import batch_rosenbrock


def test_strategy_ask():
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    for s_name, Strat in Strategies.items():
        # PBT also returns copy ID integer - treat separately
        if s_name in ["Persistent_ES", "Open_NES"]:
            popsize = 8
        else:
            popsize = 9
        if s_name != "PBT_ES":
            strategy = Strat(popsize=popsize, num_dims=2)
            params = strategy.default_params
            state = strategy.initialize(rng, params)
            x, state = strategy.ask(rng, state, params)
            assert x.shape[0] == popsize
            assert x.shape[1] == 2
    return


def test_strategy_ask_tell():
    # Loop over all strategies and test ask API
    rng = jax.random.PRNGKey(0)
    # PBT also returns copy ID integer - treat separately
    for s_name, Strat in Strategies.items():
        if s_name in ["Persistent_ES", "Open_NES"]:
            popsize = 8
        else:
            popsize = 9
        if s_name != "PBT_ES":
            strategy = Strat(popsize=popsize, num_dims=2)
            params = strategy.default_params
            state = strategy.initialize(rng, params)
            x, state = strategy.ask(rng, state, params)
            fitness = batch_rosenbrock(x, 1, 100)
            state = strategy.tell(x, fitness, state, params)
    return
