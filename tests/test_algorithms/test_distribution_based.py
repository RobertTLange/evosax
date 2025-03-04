"""Tests for distribution-based algorithms."""

import jax
import jax.numpy as jnp
from evosax.algorithms.distribution_based import distribution_based_algorithms


def test_run(
    distribution_based_algorithm_name,
    key,
    num_generations,
    population_size,
    bbob_problem,
):
    """Instantiate algo and test API."""
    # Get the algorithm class from the name
    AlgorithmClass = distribution_based_algorithms[distribution_based_algorithm_name]

    # Adjust population size for ESMC which requires odd population size
    population_size = (
        population_size + 1
        if distribution_based_algorithm_name == "ESMC"
        else population_size
    )

    # Initialize algo
    solution = bbob_problem.sample(key)
    if distribution_based_algorithm_name in ["RandomSearch"]:
        algo = AlgorithmClass(
            population_size=population_size,
            solution=solution,
            sampling_fn=bbob_problem.sample,
        )
    elif distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        num_populations = 2
        algo = AlgorithmClass(
            population_size=population_size,
            num_populations=num_populations,
            solution=solution,
        )
    else:
        algo = AlgorithmClass(population_size=population_size, solution=solution)

    # Use default parameters
    params = algo.default_params

    # Get initial mean
    if distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_populations)
        mean_init = jax.vmap(bbob_problem.sample)(keys)
    else:
        key, subkey = jax.random.split(key)
        mean_init = bbob_problem.sample(subkey)

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, mean_init, params)

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
    distribution_based_algorithm_name,
    key,
    num_generations,
    population_size,
    bbob_problem,
):
    """Instantiate algo and test API using scan."""
    # Get the algorithm class from the name
    AlgorithmClass = distribution_based_algorithms[distribution_based_algorithm_name]

    # Adjust population size for ESMC which requires odd population size
    population_size = (
        population_size + 1
        if distribution_based_algorithm_name == "ESMC"
        else population_size
    )

    # Initialize algo
    solution = bbob_problem.sample(key)
    if distribution_based_algorithm_name in ["RandomSearch"]:
        algo = AlgorithmClass(
            population_size=population_size,
            solution=solution,
            sampling_fn=bbob_problem.sample,
        )
    elif distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        num_populations = 2
        algo = AlgorithmClass(
            population_size=population_size,
            num_populations=num_populations,
            solution=solution,
        )
    else:
        algo = AlgorithmClass(population_size=population_size, solution=solution)

    # Use default parameters
    params = algo.default_params

    # Get initial mean
    if distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_populations)
        mean_init = jax.vmap(bbob_problem.sample)(keys)
    else:
        key, subkey = jax.random.split(key)
        mean_init = bbob_problem.sample(subkey)

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, mean_init, params)

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


def test_base_api(
    distribution_based_algorithm_name, key, num_dims, population_size, bbob_problem
):
    """Test the base API methods of distribution-based algorithms."""
    # Get the algorithm class from the name
    AlgorithmClass = distribution_based_algorithms[distribution_based_algorithm_name]

    # Adjust population size for ESMC which requires odd population size
    population_size = (
        17 if distribution_based_algorithm_name == "ESMC" else population_size
    )

    # Initialize algo
    solution = bbob_problem.sample(key)
    if distribution_based_algorithm_name in ["RandomSearch"]:
        algo = AlgorithmClass(
            population_size=population_size,
            solution=solution,
            sampling_fn=bbob_problem.sample,
        )
    elif distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        num_populations = 2
        algo = AlgorithmClass(
            population_size=population_size,
            num_populations=num_populations,
            solution=solution,
        )
    else:
        algo = AlgorithmClass(population_size=population_size, solution=solution)

    # Use default parameters
    params = algo.default_params

    # Get initial mean
    if distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_populations)
        mean_init = jax.vmap(bbob_problem.sample)(keys)
    else:
        key, subkey = jax.random.split(key)
        mean_init = bbob_problem.sample(subkey)

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, mean_init, params)

    # Test get_mean
    mean = algo.get_mean(state)
    if distribution_based_algorithm_name in ["SV_CMA_ES", "SV_Open_ES"]:
        assert mean.shape == (
            num_populations,
            num_dims,
        )
    else:
        assert mean.shape == (num_dims,)

    # Test metrics_fn - create a dummy population and fitness for testing
    key, subkey = jax.random.split(key)
    population = jnp.zeros((population_size, num_dims))
    fitness = jnp.zeros((population_size,))
    metrics = algo.metrics_fn(subkey, population, fitness, state, params)
    assert "best_fitness" in metrics
    assert "best_solution" in metrics
