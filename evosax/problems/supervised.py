import jax
import jax.numpy as jnp
import chex
from typing import Tuple


class SupervisedFitness(object):
    def __init__(self, problem_name: str = "MNIST", batch_size: int = 128):
        self.problem_name = problem_name
        self.batch_size = batch_size
        data = get_array_data(self.problem_name)
        self.dataloader = BatchLoader(*data, batch_size=self.batch_size)

    def set_apply_fn(self, network):
        """Set the network forward function."""
        self.network = network

    def rollout(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Evaluate a network on a supervised learning task."""
        rng_net_train, rng_net_test, rng_sample = jax.random.split(rng_input, 3)
        X_train, y_train, X_test, y_test = self.dataloader.sample(rng_sample)
        y_train_pred = self.network(
            {"params": network_params}, X_train, rng_net_train
        )
        y_test_pred = self.network(
            {"params": network_params}, X_test, rng_net_test
        )
        train_loss, train_acc = loss_and_acc(y_train_pred, y_train)
        test_loss, test_acc = loss_and_acc(y_test_pred, y_test)
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of the observation."""
        return (1,) + self.dataloader.data_shape


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    num_classes = 10
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


class BatchLoader:
    def __init__(
        self,
        X_train: chex.Array,
        y_train: chex.Array,
        X_test: chex.Array,
        y_test: chex.Array,
        batch_size: int,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.data_shape = self.X_train.shape[1:]
        self.num_train_samples = X_train.shape[0]
        self.X_test = X_test
        self.y_test = y_test
        self.num_test_samples = X_test.shape[0]
        self.batch_size = batch_size

    def sample(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Sample a single batch of X, y data."""
        train_idx = jax.random.choice(
            key,
            jnp.arange(self.num_train_samples),
            (self.batch_size,),
            replace=False,
        )
        test_idx = jax.random.choice(
            key,
            jnp.arange(self.num_test_samples),
            (self.batch_size,),
            replace=False,
        )
        return (
            jnp.take(self.X_train, train_idx, axis=0),
            jnp.take(self.y_train, train_idx, axis=0),
            jnp.take(self.X_test, test_idx, axis=0),
            jnp.take(self.y_test, test_idx, axis=0),
        )


def get_mnist_loaders():
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `SupervisedFitness` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    test_eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "~/data", download=True, train=False, transform=transform
        ),
        batch_size=10000,
        shuffle=True,
    )

    train_eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "~/data", download=True, train=True, transform=transform
        ),
        batch_size=60000,
        shuffle=True,
    )
    return test_eval_loader, train_eval_loader


def get_fashion_loaders():
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `SupervisedFitness` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_eval_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "~/data", download=True, train=False, transform=transform
        ),
        batch_size=10000,
        shuffle=True,
    )

    train_eval_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "~/data", download=True, train=True, transform=transform
        ),
        batch_size=50000,
        shuffle=True,
    )
    return test_eval_loader, train_eval_loader


def get_cifar_loaders():
    """Get PyTorch Data Loaders for CIFAR-10."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `SupervisedFitness` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="~/data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=50000, shuffle=True
    )

    testset = datasets.CIFAR10(
        root="~/data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10000, shuffle=False
    )
    return trainloader, testloader


def normalize(data_tensor):
    """re-scale image values to [-1, 1]"""
    return (data_tensor / 255.0) * 2.0 - 1.0


def get_svhn_loaders():
    """Get PyTorch Data Loaders for SVHN."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `SupervisedFitness` module."
        )

    transform = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize(x)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ]

    trainset = datasets.SVHN(
        root="~/data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=73257, shuffle=True
    )

    testset = datasets.SVHN(
        root="~/data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=26032, shuffle=False
    )
    return trainloader, testloader


def get_array_data(problem_name: str = "MNIST"):
    """Get raw data arrays to subsample from."""
    if problem_name == "MNIST":
        test_loader, train_loader = get_mnist_loaders()
    elif problem_name == "FashionMNIST":
        test_loader, train_loader = get_fashion_loaders()
    elif problem_name == "CIFAR10":
        test_loader, train_loader = get_cifar_loaders()
    elif problem_name == "SVHN":
        test_loader, train_loader = get_svhn_loaders()
    else:
        raise ValueError("Dataset is not supported.")
    for _, (train_data, train_target) in enumerate(train_loader):
        break
    for _, (test_data, test_target) in enumerate(test_loader):
        break
    return (
        jnp.array(train_data),
        jnp.array(train_target),
        jnp.array(test_data),
        jnp.array(test_target),
    )
