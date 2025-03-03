"""Tests for reinforcement learning problems."""

import jax
from evosax.problems import BraxProblem, GymnaxProblem
from evosax.problems.networks import MLP


def test_gymnax_problem_init():
    """Test GymnaxProblem initialization with default settings."""
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=5
    )

    assert problem.env_name == "CartPole-v1"
    assert problem.episode_length == 100
    assert problem.num_rollouts == 5


def test_gymnax_problem_sample():
    """Test GymnaxProblem solution sampling."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=5
    )

    # Sample a solution
    solution = problem.sample(key)

    # Check that solution is a valid PyTree
    flat_params, _ = jax.flatten_util.ravel_pytree(solution)
    assert flat_params.ndim == 1


def test_gymnax_problem_eval():
    """Test GymnaxProblem evaluation."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 2))
    problem = GymnaxProblem(
        env_name="CartPole-v1", policy=policy, episode_length=100, num_rollouts=3
    )

    # Create a batch of solutions using vmap
    population_size = 4
    keys = jax.random.split(key, population_size)

    # Create a batch of solutions
    solutions = jax.vmap(problem.sample)(keys)

    # Evaluate the solutions
    key_eval = jax.random.key(42)
    fitness, _ = problem.eval(key_eval, solutions)

    # Check shape (population_size,)
    assert fitness.shape == (population_size,)


def test_brax_problem_init():
    """Test BraxProblem initialization with default settings."""
    policy = MLP(layer_sizes=(64, 64, 1))
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=5
    )

    assert problem.env_name == "ant"
    assert problem.episode_length == 100
    assert problem.num_rollouts == 5


def test_brax_problem_sample():
    """Test BraxProblem solution sampling."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 8))  # Ant has 8 actions
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=5
    )

    # Sample a solution
    solution = problem.sample(key)

    # Check that solution is a valid PyTree
    flat_params, _ = jax.flatten_util.ravel_pytree(solution)
    assert flat_params.ndim == 1


def test_brax_problem_eval():
    """Test BraxProblem evaluation."""
    key = jax.random.key(0)
    policy = MLP(layer_sizes=(64, 64, 8))  # Ant has 8 actions
    problem = BraxProblem(
        env_name="ant", policy=policy, episode_length=100, num_rollouts=3
    )

    # Create a batch of solutions using vmap
    population_size = 4
    keys = jax.random.split(key, population_size)

    # Create a batch of solutions
    solutions = jax.vmap(problem.sample)(keys)

    # Evaluate the solutions
    key_eval = jax.random.key(42)
    fitness, _ = problem.eval(key_eval, solutions)

    # Check shape (population_size,)
    assert fitness.shape == (population_size,)
