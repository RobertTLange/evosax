"""Compatibility tests for learned evolutionary algorithms."""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evosax.algorithms.distribution_based import EvoTF_ES, LearnedES
from evosax.algorithms.population_based import LearnedGA
from evosax.learned_evolution.evotf_tools.features.fitness import get_norm_diff_best

CHECKPOINT_DIR = Path(__file__).parents[2] / "evosax" / "algorithms" / "ckpt"
CHECKPOINT_PATHS = tuple(sorted(CHECKPOINT_DIR.rglob("*.pkl")))
LEARNED_ALGORITHMS = (LearnedES, LearnedGA, EvoTF_ES)
NUM_DIMS = 2
NUM_GENERATIONS = 100
POPULATION_SIZE = 8


def test_normalized_fitness_gap_is_zero_without_previous_best():
    """The first EvoTF generation has no previous best for comparison."""
    fitness = jnp.array([1.0, 2.0, 3.0])

    result = get_norm_diff_best(fitness, best_fitness=jnp.inf)

    assert jnp.array_equal(result, jnp.zeros_like(fitness))


@pytest.mark.parametrize(
    "checkpoint_path", CHECKPOINT_PATHS, ids=lambda path: path.name
)
def test_checkpoint_uses_portable_array_serialization(checkpoint_path):
    """Bundled checkpoints must avoid version-specific array internals."""
    checkpoint = checkpoint_path.read_bytes()

    incompatible_metadata = (
        b"jax._src.array",
        b"named_shape",
        b"numpy.core",
        b"numpy._core",
    )
    assert not any(marker in checkpoint for marker in incompatible_metadata)


@pytest.mark.parametrize(
    "checkpoint_path", CHECKPOINT_PATHS, ids=lambda path: path.name
)
def test_checkpoint_stores_numeric_leaves_as_numpy_arrays(checkpoint_path):
    """Checkpoint tensors must use the portable NumPy representation."""
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)

    array_leaves = [
        leaf
        for leaf in jax.tree.leaves(checkpoint)
        if isinstance(leaf, (jax.Array, np.ndarray))
    ]
    assert array_leaves and all(isinstance(leaf, np.ndarray) for leaf in array_leaves)


@pytest.mark.parametrize(
    "algorithm_class",
    LEARNED_ALGORITHMS,
    ids=lambda algorithm_class: algorithm_class.__name__,
)
def test_learned_algorithm_runs_on_sphere(algorithm_class, key):
    """Each learned algorithm must complete a finite sphere run."""
    key, init_key = jax.random.split(key)
    algorithm, state, params, initial_best = _initialize_algorithm(
        algorithm_class, init_key
    )

    best_fitness = []
    all_values_are_finite = jnp.array(True)
    for _ in range(NUM_GENERATIONS):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = algorithm.ask(ask_key, state, params)
        fitness = jnp.sum(jnp.square(population), axis=-1)
        state, metrics = algorithm.tell(tell_key, population, fitness, state, params)
        best_fitness.append(metrics["best_fitness"])
        numeric_values = (population, fitness, *jax.tree.leaves(metrics))
        all_values_are_finite &= all(
            jnp.all(jnp.isfinite(value)) for value in numeric_values
        )

    best_fitness = jnp.asarray(best_fitness)
    is_nonincreasing = jnp.all(jnp.diff(best_fitness) <= 0)
    strictly_improves = best_fitness[-1] < initial_best
    assert all_values_are_finite & is_nonincreasing & strictly_improves


def _initialize_algorithm(algorithm_class, key):
    solution = jnp.zeros(NUM_DIMS)
    algorithm = algorithm_class(population_size=POPULATION_SIZE, solution=solution)
    params = algorithm.default_params

    if algorithm_class is LearnedGA:
        population = jax.random.normal(key, (POPULATION_SIZE, NUM_DIMS))
        fitness = jnp.sum(jnp.square(population), axis=-1)
        state = algorithm.init(key, population, fitness, params)
        initial_best = jnp.min(fitness)
    else:
        mean = jnp.ones(NUM_DIMS)
        state = algorithm.init(key, mean, params)
        initial_best = jnp.sum(jnp.square(mean))

    return algorithm, state, params, initial_best
