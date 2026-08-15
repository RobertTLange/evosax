"""Tests for distribution-based algorithms."""

import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based import (
    ASEBO,
    Open_ES,
    distribution_based_algorithms,
)


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
    distribution_based_algorithm_name,
    key,
    num_generations,
    population_size,
    bbob_problem,
):
    """Instantiate algo and test API using scan."""
    # Get the algorithm class from the name
    AlgorithmClass = distribution_based_algorithms[distribution_based_algorithm_name]

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


def test_base_api(
    distribution_based_algorithm_name, key, num_dims, population_size, bbob_problem
):
    """Test the base API methods of distribution-based algorithms."""
    # Get the algorithm class from the name
    AlgorithmClass = distribution_based_algorithms[distribution_based_algorithm_name]

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


def test_open_es_runs_with_adamw_optimizer(key, population_size, bbob_problem):
    """Open_ES should support optimizers that require the current params."""
    solution = bbob_problem.sample(key)
    algo = Open_ES(
        population_size=population_size,
        solution=solution,
        optimizer=optax.adamw(learning_rate=1e-3, weight_decay=1e-4),
    )
    params = algo.default_params

    key, subkey = jax.random.split(key)
    mean_init = bbob_problem.sample(subkey)

    key, subkey = jax.random.split(key)
    state = algo.init(subkey, mean_init, params)

    key, subkey = jax.random.split(key)
    problem_state = bbob_problem.init(subkey)

    key, key_ask, key_tell = jax.random.split(key, 3)
    population, state = algo.ask(key_ask, state, params)
    fitness, problem_state, _ = bbob_problem.eval(key_tell, population, problem_state)
    state, metrics = algo.tell(key_tell, population, fitness, state, params)

    assert jnp.all(jnp.isfinite(state.mean))
    assert jnp.isfinite(metrics["best_fitness"])


def test_asebo_uses_an_informative_active_subspace(key):
    """ASEBO should retain enough centered gradients to identify its subspace."""
    algo = ASEBO(population_size=8, solution=jnp.zeros(3), subspace_dims=1)
    params = algo.default_params
    state = algo.init(key, jnp.zeros(3), params)

    assert state.grad_subspace.shape == (2, 3)

    state = state.replace(
        generation_counter=2,
        grad_subspace=jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    _, state = algo.ask(key, state, params)

    assert jnp.allclose(state.UUT, jnp.diag(jnp.array([1.0, 0.0, 0.0])))


def test_asebo_ignores_rank_deficient_gradient_directions(key):
    """ASEBO should not use arbitrary SVD vectors from a rank-deficient archive."""
    algo = ASEBO(population_size=8, solution=jnp.zeros(3), subspace_dims=2)
    params = algo.default_params
    state = algo.init(key, jnp.zeros(3), params).replace(
        generation_counter=3,
        grad_subspace=jnp.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    _, state = algo.ask(key, state, params)

    assert jnp.allclose(state.UUT, jnp.diag(jnp.array([1.0, 0.0, 0.0])))


def test_asebo_remains_finite_after_subspace_activation(key):
    """ASEBO should continue producing finite states after activation."""
    algo = ASEBO(population_size=8, solution=jnp.zeros(3), subspace_dims=1)
    params = algo.default_params
    state = algo.init(key, jnp.zeros(3), params)

    for _ in range(4):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = algo.ask(ask_key, state, params)
        fitness = jnp.sum(population**2, axis=-1)
        state, _ = algo.tell(tell_key, population, fitness, state, params)

        assert jnp.all(jnp.isfinite(population))
        assert jnp.all(jnp.isfinite(state.mean))
        assert jnp.isfinite(state.alpha)


def test_asebo_samples_with_the_configured_standard_deviation(key):
    """ASEBO should scale active-subspace samples by the scheduled std."""

    def std_schedule(generation):
        return jnp.where(generation < 2, 1.0, 0.25)

    algo = ASEBO(
        population_size=8,
        solution=jnp.zeros(3),
        subspace_dims=1,
        std_schedule=std_schedule,
    )
    params = algo.default_params
    state = algo.init(key, jnp.zeros(3), params).replace(
        alpha=0.5,
        generation_counter=2,
        grad_subspace=jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )

    population, _ = algo.ask(key, state, params)
    decayed_state = state.replace(std=std_schedule(state.generation_counter))
    decayed_population, _ = algo.ask(key, decayed_state, params)

    assert jnp.all(jnp.isfinite(population))
    assert jnp.allclose(decayed_population, 0.25 * population, rtol=1e-5)


def test_asebo_gradient_archive_is_invariant_to_standard_deviation(key):
    """ASEBO should not reweight archived gradients when std changes."""
    algo = ASEBO(population_size=2, solution=jnp.zeros(2))
    params = algo.default_params

    def estimate_gradient(std):
        state = algo.init(key, jnp.zeros(2), params).replace(std=std)
        population = jnp.array([[std, 0.0], [-std, 0.0]])
        fitness = population[:, 0]
        state = algo._tell(key, population, fitness, state, params)
        return state.grad_subspace[-1]

    assert jnp.allclose(estimate_gradient(1.0), estimate_gradient(0.25))
