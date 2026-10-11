"""Frozen-weight and lifecycle regression tests for CMAAnchoredLES."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.distribution_based.cma_anchored_les import CMAAnchoredLES
from evosax.learned_evolution.les_tools import load_pkl_object
from flax import serialization

CHECKPOINT = (
    Path(__file__).parents[2]
    / "evosax/algorithms/ckpt/cma_anchored_les/2026_10_cma_anchored_les.pkl"
)
GOLDEN = Path(__file__).with_name("fixtures") / "cma_anchored_les.json"


def test_bundled_weights_are_original_frozen_device_parameters():
    algorithm = CMAAnchoredLES(8, jnp.zeros(3))
    leaves = jax.tree.leaves(algorithm.default_params.params)
    assert isinstance(algorithm, DistributionBasedAlgorithm)
    assert algorithm.default_params.std_init == 0.05
    assert sum(leaf.size for leaf in leaves) == 364
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    stored = load_pkl_object(CHECKPOINT)
    assert all(type(leaf) is np.ndarray for leaf in jax.tree.leaves(stored))
    assert all(leaf.dtype == np.float32 for leaf in jax.tree.leaves(stored))
    # Reconstruct the original artifact to guard every value, shape and dtype.
    original_bytes = serialization.msgpack_serialize(stored)
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "7e080b262a8def198909133facc5441aecb81664d64e3b697a6ade9d58b62295"
    )
    provenance = json.loads(CHECKPOINT.with_name("provenance.json").read_text())
    assert (
        provenance["source_checkpoint_sha256"]
        == hashlib.sha256(original_bytes).hexdigest()
    )
    assert (
        provenance["checkpoint_sha256"]
        == hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    )
    assert provenance["checkpoint_bytes"] == CHECKPOINT.stat().st_size


def test_parameter_tree_takes_precedence_over_path():
    params = load_pkl_object(CHECKPOINT)
    algorithm = CMAAnchoredLES(8, jnp.zeros(3), params, "/missing/checkpoint")
    for actual, expected in zip(
        jax.tree.leaves(algorithm.default_params.params),
        jax.tree.leaves(params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_custom_pickle_loads_same_weights():
    algorithm = CMAAnchoredLES(8, jnp.zeros(3), params_path=str(CHECKPOINT))
    default = CMAAnchoredLES(8, jnp.zeros(3))
    for actual, expected in zip(
        jax.tree.leaves(algorithm.default_params.params),
        jax.tree.leaves(default.default_params.params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", ["missing", "shape", "nan", "text", "complex"])
def test_invalid_parameter_trees_fail_at_construction(invalid):
    params = load_pkl_object(CHECKPOINT)
    layer = params["global_gates"]["params"]["Dense_0"]
    if invalid == "missing":
        del layer["bias"]
    elif invalid == "shape":
        layer["bias"] = np.zeros(99)
    elif invalid == "nan":
        layer["bias"] = np.full(8, np.nan)
    elif invalid == "text":
        layer["bias"] = np.full(8, "oops")
    else:
        layer["bias"] = np.ones(8, dtype=complex)
    with pytest.raises(ValueError, match="parameters"):
        CMAAnchoredLES(8, jnp.zeros(3), params=params)


@pytest.mark.parametrize("contents", [b"not a checkpoint", b"", b"\x80\x04"])
def test_invalid_pickle_fails_with_checkpoint_context(tmp_path, contents):
    checkpoint = tmp_path / "invalid.pkl"
    checkpoint.write_bytes(contents)
    with pytest.raises(ValueError, match="checkpoint"):
        CMAAnchoredLES(8, jnp.zeros(3), params_path=str(checkpoint))


@pytest.mark.parametrize("population_size", [1, 5, 8])
def test_pytree_sphere_run_is_finite_and_preserves_best(population_size):
    mean = {"a": jnp.ones(2), "b": jnp.ones((1, 1))}
    algorithm = CMAAnchoredLES(population_size, mean)
    params = algorithm.default_params
    state = algorithm.init(jax.random.key(0), mean, params)
    best = np.inf
    for generation in range(40):
        key = jax.random.key(generation + 1)
        population, state = algorithm.ask(key, state, params)
        fitness = jnp.sum(population["a"] ** 2, axis=1)
        fitness += jnp.sum(population["b"] ** 2, axis=(1, 2))
        state, metrics = algorithm.tell(key, population, fitness, state, params)
        assert np.isfinite(state.std).all()
        assert float(state.best_fitness) <= best
        best = float(state.best_fitness)
        assert float(metrics["best_fitness"]) == best
    assert best < 3.0
    assert algorithm.get_mean(state)["b"].shape == (1, 1)


@pytest.mark.parametrize("case_index", range(12))
def test_matches_canonical_frozen_trajectory(case_index):
    case = json.loads(GOLDEN.read_text())["cases"][case_index]
    algorithm = CMAAnchoredLES(case["population_size"], jnp.zeros(3))
    params = algorithm.default_params
    state = algorithm.init(jax.random.key(0), jnp.array([0.2, -0.3, 0.1]), params)
    state = state.replace(success_ema=jnp.full(3, case["success"]))
    if case["restart"]:
        state = state.replace(
            sigma=jnp.asarray(0.001),
            std=jnp.full(3, 0.001),
            epoch_best_fitness=-1.0,
            stall_counter=jnp.asarray(10000.0),
            best_fitness=-1.0,
            best_fitness_shaped=-1.0,
            best_solution=jnp.array([0.1, 0.2, 0.3]),
            best_solution_shaped=jnp.array([0.1, 0.2, 0.3]),
        )
    for generation, expected in enumerate(case["steps"]):
        key = jax.random.key(generation + 42)
        population, state = algorithm.ask(key, state, params)
        np.testing.assert_allclose(
            population, expected["population"], rtol=2e-5, atol=2e-7
        )
        fitness = jnp.sum((population - jnp.array([0.4, 0.1, -0.2])) ** 2, axis=1)
        state, _ = algorithm.tell(key, population, fitness, state, params)
        for field, value in expected["state"].items():
            np.testing.assert_allclose(
                getattr(state, field),
                np.asarray(value, dtype=float),
                rtol=2e-5,
                atol=2e-7,
                err_msg=f"{field}, generation {generation}",
            )


def test_custom_shaping_keeps_raw_and_shaped_incumbents_separate():
    algorithm = CMAAnchoredLES(
        4, jnp.zeros(2), fitness_shaping_fn=lambda x, f, s, p: -f
    )
    params = algorithm.default_params
    key = jax.random.key(0)
    state = algorithm.init(key, jnp.zeros(2), params)
    population = jnp.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.1], [0.5, 0.2]])
    state, metrics = algorithm.tell(key, population, jnp.arange(4.0), state, params)
    np.testing.assert_array_equal(state.best_solution, population[0])
    np.testing.assert_array_equal(state.best_solution_shaped, population[-1])
    assert float(metrics["best_fitness"]) == 0.0
    assert float(state.best_fitness_shaped) == -3.0


@pytest.mark.parametrize("sigma", [0.001, 0.04, 0.1])
def test_stall_reexpansion_respects_reference_scale_and_preserves_incumbent(sigma):
    algorithm = CMAAnchoredLES(8, jnp.zeros(3))
    params = algorithm.default_params
    key = jax.random.key(0)
    incumbent = jnp.array([0.1, 0.2, 0.3])
    state = algorithm.init(key, jnp.zeros(3), params).replace(
        sigma=jnp.asarray(sigma),
        std=jnp.full(3, sigma),
        best_solution=incumbent,
        best_solution_shaped=incumbent,
        best_fitness=-1.0,
        best_fitness_shaped=-1.0,
        epoch_best_fitness=-1.0,
    )
    population, state = algorithm.ask(key, state, params)
    ordinary, _ = algorithm.tell(key, population, jnp.zeros(8), state, params)
    stalled, _ = algorithm.tell(
        key,
        population,
        jnp.zeros(8),
        state.replace(stall_counter=jnp.asarray(10000.0)),
        params,
    )
    expected = max(
        float(ordinary.sigma), min(3 * float(ordinary.sigma), params.std_init)
    )
    np.testing.assert_allclose(stalled.sigma, expected, rtol=1e-6)
    np.testing.assert_array_equal(stalled.mean, incumbent)
    np.testing.assert_array_equal(stalled.best_solution, incumbent)
    np.testing.assert_array_equal(stalled.axis_std, np.ones(3))
    np.testing.assert_array_equal(stalled.p_sigma, np.zeros(3))
    np.testing.assert_array_equal(stalled.p_c, np.zeros(3))
    np.testing.assert_array_equal(stalled.success_ema, np.full(3, 0.5))
    assert float(stalled.stall_counter) == 0.0
    assert float(stalled.epoch_best_fitness) == np.inf
    assert float(stalled.best_fitness) == -1.0


@pytest.mark.parametrize("size", [5, 8])
@pytest.mark.parametrize("success", [0.0, 0.3, 0.5])
def test_paths_use_mass_of_pairs_actually_sampled(size, success):
    from evosax.learned_evolution.cma_anchored_les_tools import (
        initialize_meta_parameters,
    )

    key = jax.random.key(4)
    algorithm = CMAAnchoredLES(
        size, jnp.zeros(3), params=initialize_meta_parameters(key)
    )
    params = algorithm.default_params
    state = algorithm.init(key, jnp.zeros(3), params).replace(
        success_ema=jnp.full(3, success)
    )
    population, state = algorithm.ask(key, state, params)
    base = (size + 1) // 2
    fitness = jnp.arange(size, dtype=float) + 10.0
    fitness = fitness.at[0].set(0.0).at[base].set(1.0)
    result, _ = algorithm.tell(key, population, fitness, state, params)

    elite_count = size // 2
    elite_weights = np.log(elite_count + 0.5) - np.log(np.arange(elite_count) + 1)
    elite_weights /= elite_weights.sum()
    weights = np.zeros(size)
    weights[np.argsort(fitness)[:elite_count]] = elite_weights
    pair_count = size - base
    mirrored_count = int(np.ceil(np.clip(pair_count - size * success, 0, pair_count)))
    variance = np.square(weights).sum()
    variance -= 2 * sum(weights[i] * weights[base + i] for i in range(mirrored_count))
    mass = min(1 / (variance + 1e-8), size)
    weighted_noise = np.sum(
        weights[:, None] * np.asarray(population / state.sigma), axis=0
    )
    expected_path = (
        np.sqrt(algorithm.c_sigma * (2 - algorithm.c_sigma) * mass) * weighted_noise
    )
    np.testing.assert_allclose(result.p_sigma, expected_path, rtol=1e-5, atol=2e-7)
    if mirrored_count:
        assert variance < np.square(weights).sum()


def test_empty_solution_is_rejected():
    with pytest.raises(ValueError, match="nonempty solution"):
        CMAAnchoredLES(8, jnp.zeros(0))
