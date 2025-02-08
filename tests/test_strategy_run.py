
import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.core import FitnessShaper
from evosax.problems import BBOBFitness

num_iters = 25


def test_strategy_run(strategy_name):
    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    Strategy = Strategies[strategy_name]

    num_dims = 2
    x = jnp.zeros((num_dims,))

    # PBT also returns copy ID integer - treat separately
    population_size = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenAI_ES", "SV_OpenES"]:
        strategy = Strategy(npop=1, subpopulation_size=population_size, pholder_params=x)
    else:
        strategy = Strategy(population_size=population_size, pholder_params=x)
    evaluator = BBOBFitness("sphere", 2)
    fitness_shaper = FitnessShaper()

    batch_eval = evaluator.rollout
    params = strategy.default_params
    state = strategy.init(key, params)

    fitness_log = []
    for t in range(num_iters):
        key, key_ask, key_eval = jax.random.split(key, 3)
        x, state = strategy.ask(key_ask, state, params)
        fitness = batch_eval(key_eval, x)
        fitness_shaped = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
    # assert fitness[0] >= fitness[-1]


def test_strategy_scan(strategy_name):
    # Loop over all strategies and test ask API
    key = jax.random.key(0)
    Strategy = Strategies[strategy_name]

    num_dims = 2
    x = jnp.zeros((num_dims,))

    # PBT also returns copy ID integer - treat separately
    population_size = 21 if strategy_name == "ESMC" else 20
    if strategy_name in ["SV_CMA_ES", "SV_OpenAI_ES", "SV_OpenES"]:
        strategy = Strategy(npop=1, subpopulation_size=population_size, pholder_params=x)
    elif strategy_name in ["BIPOP_CMA_ES", "IPOP_CMA_ES"]:
        return
    else:
        strategy = Strategy(population_size=population_size, pholder_params=x)
    evaluator = BBOBFitness("sphere", 2)
    fitness_shaper = FitnessShaper()

    batch_eval = evaluator.rollout
    es_params = strategy.default_params

    state = strategy.init(key, es_params)

    def step(carry, _):
        """Helper function to lax.scan."""
        key, state = carry
        key, key_ask, key_eval = jax.random.split(key, 3)
        x, state = strategy.ask(key_ask, state, es_params)
        fitness = batch_eval(key_eval, x)
        fitness_shaped = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, es_params)
        return (key, state), jnp.min(fitness)

    _, best_fitness = jax.lax.scan(
        step,
        init=(key, state),
        length=num_iters,
    )
