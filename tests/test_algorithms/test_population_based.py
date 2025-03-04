"""Tests for population-based algorithms."""

import jax
import jax.numpy as jnp
from evosax.algorithms.population_based import population_based_algorithms


def test_run(
    population_based_algorithm_name, key, num_generations, population_size, bbob_problem
):
    """Instantiate strategy and test API."""
    # Get the algorithm class from the name
    AlgorithmClass = population_based_algorithms[population_based_algorithm_name]

    # Initialize the strategy
    solution = bbob_problem.sample(key)
    algo = AlgorithmClass(population_size=population_size, solution=solution)

    # Use default parameters
    params = algo.default_params

    # Get initial population and fitness
    key, key_init = jax.random.split(key)
    population_init = jnp.vstack(
        [
            bbob_problem.sample(key)
            for key in jax.random.split(key_init, population_size)
        ]
    )
    fitness_init, _ = bbob_problem.eval(key, population_init)

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    best_fitness = []
    for _ in range(num_generations):
        key, key_ask, key_tell = jax.random.split(key, 3)

        # Ask
        population, state = algo.ask(key_ask, state, params)

        # Eval using BBOB problem
        fitness, _ = bbob_problem.eval(key_tell, population)

        # Tell
        state, metrics = algo.tell(key_tell, population, fitness, state, params)

        best_fitness.append(metrics["best_fitness"])

    assert len(best_fitness) == num_generations


def test_run_scan(
    population_based_algorithm_name, key, num_generations, population_size, bbob_problem
):
    """Instantiate strategy and test API using scan."""
    # Get the algorithm class from the name
    AlgorithmClass = population_based_algorithms[population_based_algorithm_name]

    # Initialize the strategy
    solution = bbob_problem.sample(key)
    algo = AlgorithmClass(population_size=population_size, solution=solution)

    # Use default parameters
    params = algo.default_params

    # Get initial population and fitness
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, population_size)
    population_init = jax.vmap(bbob_problem.sample)(keys)

    key, subkey = jax.random.split(key)
    fitness_init, _ = bbob_problem.eval(subkey, population_init)

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    def step(carry, _):
        key, state = carry
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = algo.ask(key_ask, state, params)
        # Eval using BBOB problem
        fitness, _ = bbob_problem.eval(key_tell, population)
        state, metrics = algo.tell(key_tell, population, fitness, state, params)
        return (key, state), metrics["best_fitness"]

    _, fitness_log = jax.lax.scan(
        step,
        (key, state),
        jnp.zeros(num_generations),
    )

    assert fitness_log.shape[0] == num_generations


def test_base_api(population_based_algorithm_name, key, num_dims, population_size):
    """Test the base API methods of population-based algorithms."""
    # Get the algorithm class from the name
    AlgorithmClass = population_based_algorithms[population_based_algorithm_name]

    # Initialize the strategy
    solution = jnp.zeros((num_dims,))
    algo = AlgorithmClass(population_size=population_size, solution=solution)

    params = algo.default_params

    # Create initial population and fitness
    population_init = jnp.zeros((population_size, num_dims))
    fitness_init = jnp.zeros((population_size,))

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    # Test get_best_solution
    best_solution = algo.get_best_solution(state)
    assert best_solution.shape == (num_dims,)

    # Test get_population
    population = algo.get_population(state)
    assert population.shape == (population_size, num_dims)

    # Test metrics_fn
    key, subkey = jax.random.split(key)
    metrics = algo.metrics_fn(subkey, population_init, fitness_init, state, params)
    assert "best_fitness" in metrics
    assert "best_solution" in metrics
    assert "best_fitness_in_generation" in metrics
    assert "best_solution_in_generation" in metrics
