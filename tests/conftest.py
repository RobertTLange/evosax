"""Pytest configuration file for evosax tests."""

import jax
import jax.numpy as jnp
import pytest
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.population_based import population_based_algorithms
from evosax.problems import BBOBProblem


# Common test parameters
@pytest.fixture
def num_dims():
    return 2


@pytest.fixture
def num_generations():
    return 16


@pytest.fixture
def population_size():
    return 8


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.fixture
def bbob_problem(num_dims):
    """Create a BBOB problem instance with the sphere function."""
    return BBOBProblem(fn_name="sphere", num_dims=num_dims, seed=0)


@pytest.fixture(params=list(distribution_based_algorithms.keys()))
def distribution_based_algorithm_name(request):
    return request.param


@pytest.fixture(params=list(population_based_algorithms.keys()))
def population_based_algorithm_name(request):
    return request.param
