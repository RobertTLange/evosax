"""Tests for population-based algorithms."""

import jax
import jax.numpy as jnp
from evosax.algorithms.population_based import population_based_algorithms
from evosax.algorithms.population_based.pso import PSO


def test_pso_seeds_personal_and_global_best_from_init(key):
    """PSO must keep the best initial particle after the first ask/tell."""
    population_size = 3
    num_dims = 1
    algo = PSO(population_size=population_size, solution=jnp.zeros(num_dims))
    params = algo.default_params

    # True best at particle 2 (fitness 0); particle 0 is a distractor
    population_init = jnp.array([[10.0], [20.0], [0.0]])
    fitness_init = jnp.array([100.0, 400.0, 0.0])

    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    assert jnp.allclose(state.population_best, population_init)
    assert jnp.allclose(state.fitness_best, fitness_init)
    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert state.best_fitness == 0.0

    # First ask must use true gbest (particle 2), not particle 0
    key, key_ask, key_tell = jax.random.split(key, 3)
    _, state = algo.ask(key_ask, state, params)

    # Worse post-move positions (as under the old bogus-gbest pull)
    population = jnp.array([[10.0], [10.57], [0.57]])
    fitness = jnp.array([100.0, 105.7, 0.57])
    state, _ = algo.tell(key_tell, population, fitness, state, params)

    # Initial optimum must remain personal best for particle 2 and archive best
    assert jnp.allclose(state.population_best[2], jnp.array([0.0]))
    assert state.fitness_best[2] == 0.0
    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert state.best_fitness == 0.0


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

    key, subkey = jax.random.split(key)
    problem_state = bbob_problem.init(subkey)
    fitness_init, problem_state, _ = bbob_problem.eval(
        key, population_init, problem_state
    )

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    # Initialize problem state
    key, subkey = jax.random.split(key)
    problem_state = bbob_problem.init(subkey)

    best_fitness = []
    for _ in range(num_generations):
        key, key_ask, key_tell = jax.random.split(key, 3)

        # Ask
        population, state = algo.ask(key_ask, state, params)

        # Eval using BBOB problem
        fitness, problem_state, _ = bbob_problem.eval(
            key_tell, population, problem_state
        )

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
    problem_state = bbob_problem.init(subkey)
    fitness_init, problem_state, _ = bbob_problem.eval(
        key, population_init, problem_state
    )

    # Initialize state
    key, subkey = jax.random.split(key)
    state = algo.init(subkey, population_init, fitness_init, params)

    # Initialize problem state
    key, subkey = jax.random.split(key)
    problem_state = bbob_problem.init(subkey)

    def step(carry, _):
        key, state, problem_state = carry
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = algo.ask(key_ask, state, params)
        # Eval using BBOB problem
        fitness, problem_state, _ = bbob_problem.eval(
            key_tell, population, problem_state
        )
        state, metrics = algo.tell(key_tell, population, fitness, state, params)
        return (key, state, problem_state), metrics["best_fitness"]

    _, fitness_log = jax.lax.scan(
        step,
        (key, state, problem_state),
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
