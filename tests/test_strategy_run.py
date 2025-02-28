import jax
import jax.numpy as jnp
from evosax import Strategies
from evosax.problems import BBOBProblem

num_generations = 64
population_size = 16


def test_strategy_run(strategy_name):
    """Instantiate strategy and test API."""
    key = jax.random.key(0)
    Strategy = Strategies[strategy_name]

    num_dims = 8
    problem = BBOBProblem("sphere", num_dims)

    key, subkey = jax.random.split(key)
    solution = problem.sample(subkey)

    population_size = 17 if strategy_name == "ESMC" else 16

    if strategy_name in ["RandomSearch"]:
        es = Strategy(
            population_size=population_size,
            solution=solution,
            sampling_fn=problem.sample,
        )
    elif strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        es = Strategy(
            population_size=population_size,
            num_populations=2,
            solution=solution,
        )
    else:
        es = Strategy(population_size=population_size, solution=solution)

    params = es.default_params

    key, subkey = jax.random.split(key)
    state = es.init(subkey, params)

    fitness_log = []
    for _ in range(num_generations):
        key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
        population, state = es.ask(key_ask, state, params)
        fitness = problem.eval(key_eval, population)
        state, metrics = es.tell(key_tell, population, fitness, state, params)
        best_fitness = jnp.min(fitness)
        fitness_log.append(best_fitness)


def test_strategy_scan(strategy_name):
    """Instantiate strategy and test API using scan."""
    key = jax.random.key(0)
    Strategy = Strategies[strategy_name]

    num_dims = 8
    problem = BBOBProblem("sphere", num_dims)

    key, subkey = jax.random.split(key)
    solution = problem.sample(subkey)

    population_size = 17 if strategy_name == "ESMC" else 16

    if strategy_name in ["RandomSearch"]:
        es = Strategy(
            population_size=population_size,
            solution=solution,
            sampling_fn=problem.sample,
        )
    elif strategy_name in ["SV_CMA_ES", "SV_OpenES"]:
        es = Strategy(
            population_size=population_size,
            num_populations=2,
            solution=solution,
        )
    else:
        es = Strategy(population_size=population_size, solution=solution)

    params = es.default_params

    key, subkey = jax.random.split(key)
    state = es.init(subkey, params)

    def step(carry, _):
        key, state = carry
        key, key_ask, key_eval, key_tell = jax.random.split(key, 4)
        population, state = es.ask(key_ask, state, params)
        fitness = problem.eval(key_eval, population)
        state, metrics = es.tell(key_tell, population, fitness, state, params)
        return (key, state), jnp.min(fitness)

    _, fitness_log = jax.lax.scan(
        step,
        (key, state),
        length=num_generations,
    )
