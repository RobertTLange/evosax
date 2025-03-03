"""Tests for bbob problems."""

import jax
import jax.numpy as jnp
from evosax.problems import BBOBProblem, MetaBBOBProblem


def test_bbob_problem_init():
    """Test BBOB problem initialization with default settings."""
    problem = BBOBProblem(fn_name="sphere", num_dims=2)
    assert problem.fn_name == "sphere"
    assert problem.num_dims == 2


def test_bbob_problem_sample():
    """Test BBOB problem solution sampling."""
    key = jax.random.key(0)
    problem = BBOBProblem(fn_name="sphere", num_dims=3, seed=0)

    # Sample a solution
    solution = problem.sample(key)

    # Check shape and bounds
    assert solution.shape == (3,)
    assert jnp.all(solution >= problem.x_range[0])
    assert jnp.all(solution <= problem.x_range[1])


def test_bbob_problem_eval():
    """Test BBOB problem evaluation."""
    key = jax.random.key(0)
    problem = BBOBProblem(fn_name="sphere", num_dims=2, seed=0)

    # Create a test solution
    population_size = 5
    solution = jnp.zeros((population_size, 2))

    # Evaluate the solution
    fitness, info = problem.eval(key, solution)

    # For sphere function, origin should be close to optimal if x_opt is near origin
    assert fitness.shape == (population_size,)


def test_bbob_problem_custom_params():
    """Test BBOB problem with custom parameters."""
    x_opt = jnp.array([1.0, 1.0])
    f_opt = -10.0

    problem = BBOBProblem(
        fn_name="sphere",
        num_dims=2,
        x_opt=x_opt,
        f_opt=f_opt,
        sample_rotations=False,
        seed=0,
    )

    # Check that parameters were set correctly
    assert jnp.allclose(problem._params.x_opt, x_opt)
    assert problem._params.f_opt == f_opt
    assert jnp.allclose(problem._params.R, jnp.eye(2))
    assert jnp.allclose(problem._params.Q, jnp.eye(2))


def test_meta_bbob_problem():
    """Test MetaBBOBProblem initialization and parameter sampling."""
    key = jax.random.key(0)

    # Initialize meta problem with multiple functions
    meta_problem = MetaBBOBProblem(
        fn_names=["sphere", "rastrigin", "rosenbrock"],
        min_num_dims=2,
        max_num_dims=5,
        noise_config={
            "noise_model_names": "noiseless",
            "use_stabilization": True,
        },
    )

    # Sample parameters
    params = meta_problem.sample_params(key)

    # Check parameter shapes and values
    assert params.num_dims >= 2 and params.num_dims <= 5
    assert params.x_opt.shape == (5,)  # max_num_dims
    assert params.R.shape == (5, 5)  # max_num_dims x max_num_dims
    assert params.Q.shape == (5, 5)  # max_num_dims x max_num_dims

    # Initialize state
    state = meta_problem.init(key, params)
    assert state.counter == 0

    # Test solution sampling
    solution = meta_problem.sample(key)
    assert solution.shape == (5,)  # max_num_dims
    assert jnp.all(solution >= meta_problem.x_range[0])
    assert jnp.all(solution <= meta_problem.x_range[1])

    # Test evaluation
    population_size = 3
    solutions = jnp.zeros((population_size, 5))  # population_size x max_num_dims
    fitness, new_state, _ = meta_problem.eval(key, solutions, state, params)

    assert fitness.shape == (population_size,)
    assert new_state.counter == state.counter + 1


def test_meta_bbob_rotation_matrix():
    """Test the random rotation matrix generation in MetaBBOBProblem."""
    key = jax.random.key(0)
    meta_problem = MetaBBOBProblem(fn_names=["sphere"], min_num_dims=3, max_num_dims=3)

    # Generate rotation matrix
    rotation = meta_problem.generate_random_rotation(key, 3)

    # Check properties of rotation matrix
    assert rotation.shape == (3, 3)

    # Check orthogonality: R^T R should be close to identity
    assert jnp.allclose(rotation.T @ rotation, jnp.eye(3), atol=1e-5)

    # Check determinant is 1 (proper rotation)
    assert jnp.allclose(jnp.linalg.det(rotation), 1.0, atol=1e-5)
