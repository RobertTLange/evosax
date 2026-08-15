"""Tests for population-based algorithms."""

import jax
import jax.numpy as jnp
from evosax.algorithms import DifferentialEvolution
from evosax.algorithms.population_based import population_based_algorithms
from evosax.algorithms.population_based.differential_evolution import Params
from evosax.algorithms.population_based.pso import PSO
from evosax.core.fitness_shaping import centered_rank_fitness_shaping_fn


def _fitness_relative_to_archive(population, fitness, state, params):
    return fitness - state.best_fitness


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


def test_pso_ignores_nan_initial_fitness_for_archive_and_personal_best(key):
    """Finite initial candidates must survive alongside failed evaluations."""
    algo = PSO(population_size=2, solution=jnp.zeros(1))
    population = jnp.array([[10.0], [0.0]])
    fitness = jnp.array([jnp.nan, 0.0])

    state = algo.init(key, population, fitness, algo.default_params)

    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert state.best_fitness == 0.0
    assert jnp.isinf(state.fitness_best[0])
    assert state.fitness_best[1] == 0.0


def test_population_archive_recovers_after_all_nan_initial_fitness(key):
    """A later finite evaluation replaces an initially empty archive."""
    algo = PSO(population_size=2, solution=jnp.zeros(1))
    initial_population = jnp.array([[10.0], [20.0]])
    initial_fitness = jnp.array([jnp.nan, jnp.nan])
    state = algo.init(key, initial_population, initial_fitness, algo.default_params)

    population = jnp.array([[10.0], [0.0]])
    fitness = jnp.array([jnp.nan, 0.0])
    state, _ = algo.tell(key, population, fitness, state, algo.default_params)

    assert jnp.allclose(state.best_solution, jnp.array([0.0]))
    assert state.best_fitness == 0.0


def test_pso_does_not_rank_failed_initial_evaluations_as_personal_bests(key):
    """Fitness shaping cannot turn failed evaluations into valid personal bests."""
    algo = PSO(
        population_size=2,
        solution=jnp.zeros(1),
        fitness_shaping_fn=centered_rank_fitness_shaping_fn,
    )
    population = jnp.array([[10.0], [20.0]])
    fitness = jnp.array([jnp.nan, jnp.nan])

    state = algo.init(key, population, fitness, algo.default_params)

    assert jnp.all(jnp.isinf(state.fitness))
    assert jnp.all(jnp.isinf(state.fitness_best))

    population = jnp.array([[10.0], [0.0]])
    fitness = jnp.array([jnp.nan, 0.0])
    state, _ = algo.tell(key, population, fitness, state, algo.default_params)

    assert jnp.isinf(state.fitness_best[0])
    assert jnp.isfinite(state.fitness_best[1])


def test_population_init_shaping_receives_seeded_archive(key):
    """Initial shaping sees the same archive state as tell-time shaping."""
    algo = PSO(
        population_size=2,
        solution=jnp.zeros(1),
        fitness_shaping_fn=_fitness_relative_to_archive,
    )
    population = jnp.array([[1.0], [2.0]])
    fitness = jnp.array([3.0, 4.0])

    state = algo.init(key, population, fitness, algo.default_params)

    assert jnp.allclose(state.fitness, jnp.array([0.0, 1.0]))
    assert jnp.allclose(state.fitness_best, jnp.array([0.0, 1.0]))


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


def test_differential_evolution_default_dithering_inactive(key):
    """Test default dithering params preserve fixed-weight behavior."""
    solution = jnp.zeros((2,))
    algo = DifferentialEvolution(population_size=6, solution=solution)
    params = algo.default_params.replace(crossover_rate=1.0)

    population = jnp.arange(12, dtype=float).reshape(6, 2)
    fitness = jnp.arange(6, dtype=float)

    key, key_init, key_ask = jax.random.split(key, 3)
    state = algo.init(key_init, population, fitness, params)

    fixed_params = params.replace(
        differential_weight_min=0.0,
        differential_weight_max=0.0,
    )
    population_default, _ = algo.ask(key_ask, state, params)
    population_fixed, _ = algo.ask(key_ask, state, fixed_params)

    assert jnp.allclose(population_default, population_fixed)


def test_differential_evolution_default_params_preserve_seeded_candidates():
    """Test defaults retain the pre-dithering candidate stream."""
    algo = DifferentialEvolution(population_size=6, solution=jnp.zeros((2,)))
    params = algo.default_params.replace(crossover_rate=1.0)
    population = jnp.arange(12, dtype=float).reshape(6, 2)
    fitness = jnp.arange(6, dtype=float)
    _, key_init, key_ask = jax.random.split(jax.random.key(0), 3)
    state = algo.init(key_init, population, fitness, params)

    candidates, _ = algo.ask(key_ask, state, params)

    expected_candidates = jnp.array(
        [
            [-4.8, -3.8],
            [-1.6, -0.6],
            [3.2, 4.2],
            [-6.4, -5.4],
            [-1.6, -0.6],
            [-3.2, -2.2],
        ]
    )
    assert jnp.allclose(candidates, expected_candidates)


def test_differential_evolution_params_support_legacy_construction():
    """Test fixed-weight Params construction remains compatible."""
    params = Params(
        elitism=True,
        crossover_rate=0.9,
        differential_weight=0.8,
    )

    assert params.differential_weight == 0.8


def test_differential_evolution_dithering_uses_dedicated_key_stream():
    """Test dithering does not consume a member's random key stream."""
    algo = DifferentialEvolution(population_size=6, solution=jnp.zeros((2,)))
    params = algo.default_params.replace(
        crossover_rate=1.0,
        differential_weight_min=0.0,
        differential_weight_max=1.0,
    )
    population = jnp.arange(12, dtype=float).reshape(6, 2)
    fitness = jnp.arange(6, dtype=float)
    _, key_init, key_ask = jax.random.split(jax.random.key(0), 3)
    state = algo.init(key_init, population, fitness, params)

    candidates, _ = algo.ask(key_ask, state, params)

    expected_candidates = jnp.array(
        [
            [-0.92328, 0.07672],
            [-0.30776, 0.69224],
            [0.61552, 1.61552],
            [-1.23104, -0.23104],
            [-0.30776, 0.69224],
            [-0.61552, 0.38448],
        ]
    )
    assert jnp.allclose(candidates, expected_candidates)


def test_differential_evolution_dithering_changes_population(key):
    """Test active dithering range affects generated candidates."""
    solution = jnp.zeros((2,))
    algo = DifferentialEvolution(population_size=6, solution=solution)
    params = algo.default_params.replace(
        crossover_rate=1.0,
        differential_weight=0.0,
        differential_weight_min=0.0,
        differential_weight_max=0.0,
    )
    dither_params = params.replace(
        differential_weight_min=1.0,
        differential_weight_max=1.000001,
    )

    population = jnp.arange(12, dtype=float).reshape(6, 2)
    fitness = jnp.arange(6, dtype=float)

    key, key_init, key_ask = jax.random.split(key, 3)
    state = algo.init(key_init, population, fitness, params)

    population_fixed, _ = algo.ask(key_ask, state, params)
    population_dithered, _ = algo.ask(key_ask, state, dither_params)

    assert population_dithered.shape == population.shape
    assert jnp.all(jnp.isfinite(population_dithered))
    assert not jnp.allclose(population_dithered, population_fixed)


def test_differential_evolution_dithering_scan(key):
    """Test dithered DifferentialEvolution inside a scan loop."""
    solution = jnp.zeros((2,))
    algo = DifferentialEvolution(population_size=6, solution=solution)
    params = algo.default_params.replace(
        differential_weight_min=0.5,
        differential_weight_max=1.0,
    )

    population = jnp.arange(12, dtype=float).reshape(6, 2) / 10
    fitness = jnp.sum(jnp.square(population), axis=-1)

    key, key_init = jax.random.split(key)
    state = algo.init(key_init, population, fitness, params)

    def step(carry, _):
        key, state = carry
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = algo.ask(key_ask, state, params)
        fitness = jnp.sum(jnp.square(population), axis=-1)
        state, metrics = algo.tell(key_tell, population, fitness, state, params)
        return (key, state), metrics["best_fitness"]

    _, fitness_log = jax.lax.scan(step, (key, state), jnp.zeros(4))

    assert fitness_log.shape == (4,)
    assert jnp.all(jnp.isfinite(fitness_log))
