import jax
import jax.numpy as jnp

from ..types import PyTree


class VisionProblem:
    def __init__(
        self,
        task_name: str = "MNIST",
        batch_size: int = 1024,
        test: bool = False,
    ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.steps_per_member = 1
        self.test = test
        self.num_classes = 10
        self.action_shape = 10
        data = get_array_data(self.task_name, self.test)
        self.dataloader = BatchLoader(*data, batch_size=self.batch_size)

    def set_apply_fn(self, network):
        """Set the network forward function."""
        self.network = network
        self.eval_pop = jax.vmap(self.eval_ffw, in_axes=(None, 0))
        self.eval = jax.jit(self.eval_vmap)

    def eval_vmap(self, key: jax.Array, network_params: PyTree):
        """Vectorize evaluation. Reshape output correctly."""
        loss, acc = self.eval_pop(key, network_params)
        loss_re = loss.reshape(-1, 1)
        acc_re = acc.reshape(-1, 1)
        return loss_re, acc_re

    def eval_ffw(self, key: jax.Array, network_params: PyTree) -> PyTree:
        """Evaluate a network on a supervised learning task."""
        key_sample, key_network = jax.random.split(key)
        X, y = self.dataloader.sample(key_sample)
        y_pred = self.network(network_params, X, key_network)
        loss, acc = loss_and_acc(y_pred, y, self.num_classes)
        # Return negative loss to maximize!
        return -1 * loss, acc

    @property
    def input_shape(self) -> tuple[int]:
        """Get the shape of the observation."""
        return (1,) + self.dataloader.data_shape


def loss_and_acc(
    y_pred: jax.Array, y_true: jax.Array, num_classes: int
) -> tuple[jax.Array, jax.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


class BatchLoader:
    def __init__(
        self,
        X: jax.Array,
        y: jax.Array,
        batch_size: int,
    ):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Sample a single batch of X, y data."""
        sample_idx = jax.random.choice(
            key,
            jnp.arange(self.num_train_samples),
            (self.batch_size,),
            replace=False,
        )
        return (
            jnp.take(self.X, sample_idx, axis=0),
            jnp.take(self.y, sample_idx, axis=0),
        )


def get_mnist_loaders(test: bool = False):
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionProblem` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    bs = 10000 if test else 60000
    loader = torch.utils.data.DataLoader(
        datasets.MNIST("~/data", download=True, train=not test, transform=transform),
        batch_size=bs,
        shuffle=False,
    )
    return loader


def get_fashion_loaders(test: bool = False):
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionProblem` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    bs = 10000 if test else 60000
    loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "~/data", download=True, train=not test, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )
    return loader


def get_cifar_loaders(test: bool = False):
    """Get PyTorch Data Loaders for CIFAR-10."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionProblem` module."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    bs = 10000 if test else 50000
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root="~/data", train=not test, download=True, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )
    return loader


def normalize(data_tensor):
    """re-scale image values to [-1, 1]"""
    return (data_tensor / 255.0) * 2.0 - 1.0


def get_svhn_loaders(test: bool = False):
    """Get PyTorch Data Loaders for SVHN."""
    try:
        import torch
        from torchvision import datasets, transforms
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"{err}. You need to install `torch` and `torchvision`"
            "to use the `VisionProblem` module."
        )

    transform = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: normalize(x)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    ]

    bs = 26032 if test else 73257
    loader = torch.utils.data.DataLoader(
        datasets.SVHN(
            root="~/data", train=not test, download=True, transform=transform
        ),
        batch_size=bs,
        shuffle=False,
    )
    return loader


def get_array_data(task_name: str = "MNIST", test: bool = False):
    """Get raw data arrays to subsample from."""
    if task_name == "MNIST":
        loader = get_mnist_loaders(test)
    elif task_name == "FashionMNIST":
        loader = get_fashion_loaders(test)
    elif task_name == "CIFAR10":
        loader = get_cifar_loaders(test)
    elif task_name == "SVHN":
        loader = get_svhn_loaders(test)
    else:
        raise ValueError("Dataset is not supported.")
    for _, (data, target) in enumerate(loader):
        break
    return jnp.array(data), jnp.array(target)
