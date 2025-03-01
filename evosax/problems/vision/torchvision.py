"""TorchVision Problem for Computer Vision Optimization.

This module implements a problem class for computer vision optimization using
the TorchVision library, which provides access to common image datasets.

The TorchVisionProblem class handles:
- Dataset loading and preprocessing
- Batch sampling for network training/evaluation
- Network evaluation on classification tasks
- Performance metrics calculation (loss and accuracy)

Supported datasets include MNIST, FashionMNIST, CIFAR10, and SVHN.

[1] https://pytorch.org/vision/stable/index.html
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from ...types import PyTree, Solution


class TorchVisionProblem:
    """TorchVision Problem for Computer Vision Optimization."""

    def __init__(
        self,
        task_name: str,
        network: nn.Module,
        batch_size: int = 1024,
    ):
        """Initialize the TorchVisionProblem."""
        self.task_name = task_name
        self.network = network
        self.batch_size = batch_size

        if self.task_name == "MNIST":
            self.get_dataset = self.get_mnist
        elif self.task_name == "FashionMNIST":
            self.get_dataset = self.get_fashion_mnist
        elif self.task_name == "CIFAR10":
            self.get_dataset = self.get_cifar10
        elif self.task_name == "SVHN":
            self.get_dataset = self.get_svhn
        else:
            raise ValueError(f"Dataset {self.task_name} is not supported.")

        # Get datasets and dataloaders
        self.dataset_train, self.dataset_test, self.loader_train, self.loader_test = (
            self.get_dataset()
        )

        # Put train data on device
        for image, target in self.loader_train:
            break
        self.image_train, self.target_train = jnp.array(image), jnp.array(target)

        # Put test data on device
        for image, target in self.loader_test:
            break
        self.image_test, self.target_test = jnp.array(image), jnp.array(target)

    @property
    def input_shape(self) -> tuple[int]:
        """Get image shape."""
        return self.dataloader.data_shape

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.dataset_train.classes)

    @partial(jax.jit, static_argnames=("self",))
    def sample_batch(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample a batch of data."""
        x = jax.random.choice(key, self.image_train, (self.batch_size,), replace=False)
        y = jax.random.choice(key, self.target_train, (self.batch_size,), replace=False)
        return x, y

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key: jax.Array, network_params: PyTree):
        """Evaluate a population of networks."""
        # Pegasus trick
        loss, acc = jax.vmap(self._predict, in_axes=(None, 0))(key, network_params)
        return loss, acc

    def _predict(self, key: jax.Array, network_params: PyTree) -> PyTree:
        """Evaluate network params on a batch."""
        key_sample, key_network = jax.random.split(key)

        # Sample batch
        x, y = self.sample_batch(key_sample)

        # Predict
        y_pred = self.network.apply(network_params, x, key_network)

        # Calculate accuracy
        acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y)

        # Softmax cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)

        # Take the mean over the batch
        loss = jnp.mean(loss)

        return loss, acc

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        key_init, key_sample, key_input = jax.random.split(key, 3)
        x, y = self.sample_batch(key_sample)
        return self.network.init(key_init, x, key_input)

    def get_mnist(self):
        """Get the MNIST dataset."""
        try:
            import torch
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError(
                "You need to install `torchvision` to use this problem class."
            )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        # Load the MNIST dataset
        dataset_train = datasets.MNIST(
            "~/data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.MNIST(
            "~/data", download=True, train=False, transform=transform
        )

        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=len(dataset_train), shuffle=False
        )
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=len(dataset_test), shuffle=False
        )

        return dataset_train, dataset_test, loader_train, loader_test

    def get_fashion_mnist(self):
        """Get the Fashion MNIST dataset."""
        try:
            import torch
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError(
                "You need to install `torchvision` to use this problem class."
            )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        # Load the Fashion MNIST dataset
        dataset_train = datasets.FashionMNIST(
            "~/data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.FashionMNIST(
            "~/data", download=True, train=False, transform=transform
        )

        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=len(dataset_train), shuffle=False
        )
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=len(dataset_test), shuffle=False
        )

        return dataset_train, dataset_test, loader_train, loader_test

    def get_cifar10(self):
        """Get the CIFAR-10 dataset."""
        try:
            import torch
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError(
                "You need to install `torchvision` to use this problem class."
            )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
                ),
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        # Load the CIFAR-10 dataset
        dataset_train = datasets.CIFAR10(
            "~/data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.CIFAR10(
            "~/data", download=True, train=False, transform=transform
        )

        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=len(dataset_train), shuffle=False
        )
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=len(dataset_test), shuffle=False
        )

        return dataset_train, dataset_test, loader_train, loader_test

    def get_svhn(self):
        """Get the SVHN dataset."""
        try:
            import torch
            from torchvision import datasets, transforms
        except ImportError:
            raise ImportError(
                "You need to install `torchvision` to use this problem class."
            )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * (x / 255) - 1.0),
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        # Load the SVHN dataset
        dataset_train = datasets.SVHN(
            "~/data", download=True, split="train", transform=transform
        )
        dataset_test = datasets.SVHN(
            "~/data", download=True, split="test", transform=transform
        )

        # Create data loaders
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=len(dataset_train), shuffle=False
        )
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=len(dataset_test), shuffle=False
        )

        return dataset_train, dataset_test, loader_train, loader_test
