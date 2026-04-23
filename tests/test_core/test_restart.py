"""Tests for restart condition functions."""

import jax
import jax.numpy as jnp
from evosax.core.restart import FitnessStdRestartParams, fitness_std_cond


def test_fitness_std_cond_relative_tolerance_converges():
    """Test convergence using relative tolerance."""
    fitness = jnp.array([999.9, 1000.0, 1000.1])
    restart_params = FitnessStdRestartParams(tol=0.001)

    converged = fitness_std_cond(None, fitness, None, None, None, restart_params)

    assert bool(converged)


def test_fitness_std_cond_relative_tolerance_not_converged():
    """Test non-convergence when fitness spread is too large."""
    fitness = jnp.array([990.0, 1000.0, 1010.0])
    restart_params = FitnessStdRestartParams(tol=0.001)

    converged = fitness_std_cond(None, fitness, None, None, None, restart_params)

    assert not bool(converged)


def test_fitness_std_cond_absolute_tolerance():
    """Test convergence near zero mean using absolute tolerance."""
    fitness = jnp.array([-0.05, 0.0, 0.05])
    restart_params = FitnessStdRestartParams(tol=0.0, atol=0.1)

    converged = fitness_std_cond(None, fitness, None, None, None, restart_params)

    assert bool(converged)


def test_fitness_std_cond_nonfinite_not_converged():
    """Test non-finite fitness values do not converge."""
    fitness = jnp.array([1.0, 1.0, jnp.nan])
    restart_params = FitnessStdRestartParams(tol=0.001)

    converged = fitness_std_cond(None, fitness, None, None, None, restart_params)

    assert not bool(converged)


def test_fitness_std_cond_jit():
    """Test condition is compatible with JIT."""
    restart_params = FitnessStdRestartParams(tol=0.001)

    @jax.jit
    def is_converged(fitness):
        return fitness_std_cond(None, fitness, None, None, None, restart_params)

    fitness = jnp.array([999.9, 1000.0, 1000.1])

    assert bool(is_converged(fitness))
