"""Regression tests for Stein variational exploration schedules."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from evosax.algorithms import SV_Open_ES


@pytest.mark.parametrize(
    "schedule",
    [optax.constant_schedule(0.5), optax.exponential_decay(1.0, 1, 0.5)],
    ids=["constant", "decaying"],
)
def test_sv_open_es_updates_exploration_schedule(schedule):
    """Each population's next samples must use the configured std schedule."""
    algorithm = SV_Open_ES(
        population_size=4,
        num_populations=2,
        solution=jnp.zeros(2),
        std_schedule=schedule,
    )
    params = algorithm.default_params
    key = jax.random.key(0)
    state = algorithm.init(key, jnp.array([[0.2, -0.3], [0.5, 1.0]]), params)

    def step(state, _):
        population, state = algorithm.ask(key, state, params)
        fitness = jnp.sum(population**2, axis=-1)
        state, _ = algorithm.tell(key, population, fitness, state, params)
        return state, state.std

    state, standard_deviations = jax.lax.scan(step, state, jnp.arange(3))
    expected = jnp.array([schedule(generation) for generation in range(3)])
    np.testing.assert_allclose(
        standard_deviations, jnp.repeat(expected[:, None], 2, axis=1)
    )
    assert state.std.shape == (2,)

    population, _ = algorithm.ask(key, state, params)
    unit_population, _ = algorithm.ask(
        key, state.replace(std=jnp.ones_like(state.std)), params
    )
    centered = population.reshape(2, 4, 2) - state.mean[:, None, :]
    unit_centered = unit_population.reshape(2, 4, 2) - state.mean[:, None, :]
    np.testing.assert_allclose(
        centered, unit_centered * expected[-1], rtol=1e-5, atol=1e-6
    )
