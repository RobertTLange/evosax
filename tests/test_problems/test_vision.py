"""Tests for vision problems."""

import jax
import jax.numpy as jnp
from evosax.problems import TorchVisionProblem
from evosax.problems.networks import CNN, identity_output_fn


def test_torchvision_problem_init():
    """Test TorchVisionProblem initialization with default settings."""
    # Define a simple CNN for MNIST
    network = CNN(
        num_filters=[16, 32],
        kernel_sizes=[(3, 3), (3, 3)],
        strides=[(1, 1), (1, 1)],
        mlp_layer_sizes=[64, 10],
        output_fn=identity_output_fn,
    )

    problem = TorchVisionProblem(
        task_name="MNIST",
        network=network,
        batch_size=128,
    )

    assert problem.task_name == "MNIST"
    assert problem.batch_size == 128


def test_torchvision_problem_sample():
    """Test TorchVisionProblem solution sampling."""
    key = jax.random.key(0)

    # Define a simple CNN for MNIST
    network = CNN(
        num_filters=[16, 32],
        kernel_sizes=[(3, 3), (3, 3)],
        strides=[(1, 1), (1, 1)],
        mlp_layer_sizes=[64, 10],
        output_fn=identity_output_fn,
    )

    problem = TorchVisionProblem(
        task_name="MNIST",
        network=network,
        batch_size=128,
    )

    # Sample a solution
    solution = problem.sample(key)

    # Check that solution is a valid PyTree
    flat_params, _ = jax.flatten_util.ravel_pytree(solution)
    assert flat_params.ndim == 1


def test_torchvision_problem_batch_sampling():
    """Test batch sampling in TorchVisionProblem."""
    key = jax.random.key(0)

    # Define a simple CNN for MNIST
    network = CNN(
        num_filters=[16, 32],
        kernel_sizes=[(3, 3), (3, 3)],
        strides=[(1, 1), (1, 1)],
        mlp_layer_sizes=[64, 10],
        output_fn=identity_output_fn,
    )

    problem = TorchVisionProblem(
        task_name="MNIST",
        network=network,
        batch_size=128,
    )

    # Sample a batch
    x, y = problem.sample_batch(key)

    # Check shapes
    assert x.shape[0] == 128  # batch_size
    assert y.shape[0] == 128  # batch_size
    assert x.ndim == 4  # (batch_size, height, width, channels)


def test_torchvision_problem_eval():
    """Test TorchVisionProblem evaluation."""
    key = jax.random.key(0)

    # Define a simple CNN for MNIST
    network = CNN(
        num_filters=[16, 32],
        kernel_sizes=[(3, 3), (3, 3)],
        strides=[(1, 1), (1, 1)],
        mlp_layer_sizes=[64, 10],
        output_fn=identity_output_fn,
    )

    problem = TorchVisionProblem(
        task_name="MNIST",
        network=network,
        batch_size=128,
    )

    # Create a batch of solutions using vmap
    population_size = 2
    keys = jax.random.split(key, population_size)
    solutions = jax.vmap(problem.sample)(keys)

    # Evaluate the solutions
    key_eval = jax.random.key(42)
    loss, acc = problem.eval(key_eval, solutions)

    # Check shapes
    assert loss.shape == (population_size,)
    assert acc.shape == (population_size,)
    assert jnp.all(loss >= 0.0)  # Loss should be non-negative
    assert jnp.all((acc >= 0.0) & (acc <= 1.0))  # Accuracy between 0 and 1


def test_different_datasets():
    """Test TorchVisionProblem with different datasets."""
    # Define a simple CNN
    network = CNN(
        num_filters=[16, 32],
        kernel_sizes=[(3, 3), (3, 3)],
        strides=[(1, 1), (1, 1)],
        mlp_layer_sizes=[64, 10],
        output_fn=identity_output_fn,
    )

    # Test with different datasets
    datasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

    for dataset in datasets:
        problem = TorchVisionProblem(
            task_name=dataset,
            network=network,
            batch_size=64,
        )

        assert problem.task_name == dataset

        # Basic functionality test
        key = jax.random.key(0)
        x, y = problem.sample_batch(key)

        assert x.shape[0] == 64  # batch_size
        assert y.shape[0] == 64  # batch_size
