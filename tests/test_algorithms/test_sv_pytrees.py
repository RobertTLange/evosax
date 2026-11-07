"""Regression tests for structured Stein variational populations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evosax.algorithms import SV_CMA_ES, SV_Open_ES
from jax.flatten_util import ravel_pytree


@pytest.mark.parametrize("algorithm", [SV_Open_ES, SV_CMA_ES])
@pytest.mark.parametrize("scalar_leaf", [False, True])
def test_sv_pytree_round_trip_matches_flat_population(algorithm, scalar_leaf):
    """Structured candidates must keep their leaf order in ask/tell updates."""
    solution = {
        "bias": jnp.array([0.25, -0.5]),
        "kernel": jnp.arange(4.0).reshape(2, 2) / 10,
    }
    if scalar_leaf:
        solution["scale"] = jnp.array(0.1)
    flat_solution, _ = ravel_pytree(solution)
    structured = algorithm(population_size=4, num_populations=2, solution=solution)
    flat = algorithm(population_size=4, num_populations=2, solution=flat_solution)
    params = structured.default_params
    means = jax.tree.map(lambda leaf: jnp.stack([leaf, leaf + 0.5]), solution)
    flat_means = jax.vmap(lambda tree: ravel_pytree(tree)[0])(means)
    key = jax.random.key(0)
    state = structured.init(key, means, params)
    flat_state = flat.init(key, flat_means, params)

    population, state = structured.ask(key, state, params)
    flat_population, flat_state = flat.ask(key, flat_state, params)
    flattened = jax.vmap(lambda tree: ravel_pytree(tree)[0])(population)
    np.testing.assert_allclose(flattened, flat_population)
    fitness = jnp.sum(flat_population**2, axis=-1)
    state, metrics = structured.tell(key, population, fitness, state, params)
    flat_state, flat_metrics = flat.tell(
        key, flat_population, fitness, flat_state, params
    )

    for actual, expected in zip(
        jax.tree.leaves((state, metrics)),
        jax.tree.leaves((flat_state, flat_metrics)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
