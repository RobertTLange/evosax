"""Tests for learned-evolution utility functions."""

import base64
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from evosax.learned_evolution.les_tools import (
    device_put_parameter_arrays,
    load_pkl_object,
    norm_diff_best,
)

LEGACY_JAX_ARRAY_PICKLE = base64.b64decode(
    "gASV6AAAAAAAAAB9lIwHd2VpZ2h0c5SMDmpheC5fc3JjLmFycmF5lIwS"
    "X3JlY29uc3RydWN0X2FycmF5lJOUKIwVbnVtcHkuY29yZS5tdWx0aWFy"
    "cmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RL"
    "AIWUQwFilIeUKEsBSwKFlGgIjAVkdHlwZZSTlIwCZjSUiYiHlFKUKEsD"
    "jAE8lE5OTkr/////Sv////9LAHSUYolDCAAAgD8AAABAlHSUfZQojAl3"
    "ZWFrX3R5cGWUiYwLbmFtZWRfc2hhcGWUfZR1dJRSlHMu"
)


def test_norm_diff_best_normalizes_fitness_gap() -> None:
    fitness = jnp.array([2.0, 3.0, 5.0])

    result = norm_diff_best(fitness, best_fitness=1.0)

    assert jnp.allclose(result, jnp.array([1 / 3, 2 / 3, 1.0]))


def test_load_pkl_object_restores_legacy_jax_arrays(tmp_path) -> None:
    expected = np.array([1.0, 2.0], dtype=np.float32)
    checkpoint_path = tmp_path / "legacy.pkl"
    checkpoint_path.write_bytes(LEGACY_JAX_ARRAY_PICKLE)

    checkpoint = load_pkl_object(checkpoint_path)

    weights = checkpoint["weights"]
    assert isinstance(weights, np.ndarray) and np.array_equal(weights, expected)


def test_load_pkl_object_restores_legacy_jax_arrays_from_bytes() -> None:
    checkpoint = load_pkl_object(LEGACY_JAX_ARRAY_PICKLE, pkg_load=True)

    weights = checkpoint["weights"]
    assert isinstance(weights, np.ndarray) and np.array_equal(weights, [1.0, 2.0])


def test_load_pkl_object_preserves_current_jax_arrays() -> None:
    checkpoint_bytes = pickle.dumps({"weights": jnp.array([1.0, 2.0])})

    checkpoint = load_pkl_object(checkpoint_bytes, pkg_load=True)

    assert isinstance(checkpoint["weights"], jax.Array)


def test_legacy_checkpoint_fixture_targets_removed_jax_reducer() -> None:
    expected_globals = (b"jax._src.array", b"_reconstruct_array", b"named_shape")

    assert all(value in LEGACY_JAX_ARRAY_PICKLE for value in expected_globals)


def test_device_put_parameter_arrays_preserves_jax_arrays() -> None:
    parameters = {"weights": jnp.array([1.0, 2.0])}

    placed = device_put_parameter_arrays(parameters)

    assert placed["weights"] is parameters["weights"]


def test_device_put_parameter_arrays_places_numpy_scalars() -> None:
    parameters = {"scale": np.float32(1.0)}

    placed = device_put_parameter_arrays(parameters)

    assert isinstance(placed["scale"], jax.Array)
