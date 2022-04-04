import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional


class VisionFitness(object):
    def __init__(
        self,
        task_name: str = "MNIST",
        batch_size: int = 1024,
        test: bool = False,
        n_devices: Optional[int] = None,
    ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.steps_per_member = 1
        self.test = test
        self.num_classes = 10
        self.action_shape = 10
        data = get_array_data(self.task_name, self.test)
        self.dataloader = BatchLoader(*data, batch_size=self.batch_size)
        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

    def set_apply_fn(self, map_dict, network):
        """Set the network forward function."""
        self.network = network
        self.rollout_pop = jax.vmap(self.rollout_ffw, in_axes=(None, map_dict))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout = self.rollout_pmap
            print(
                f"VisionFitness: {self.n_devices} devices detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )
        else:
            self.rollout = jax.jit(self.rollout_vmap)

    def rollout_vmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Vectorize rollout. Reshape output correctly."""
        loss, acc = self.rollout_pop(rng_input, network_params)
        loss_re = loss.reshape(-1, 1)
        acc_re = acc.reshape(-1, 1)
        return loss_re, acc_re

    def rollout_pmap(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1))
        loss_dev, acc_dev = jax.pmap(self.rollout_pop)(
            keys_pmap, network_params
        )
        loss_re = loss_dev.reshape(-1, 1)
        acc_re = acc_dev.reshape(-1, 1)
        return loss_re, acc_re

    def rollout_ffw(
        self, rng_input: chex.PRNGKey, network_params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """Evaluate a network on a supervised learning task."""
        rng_net, rng_sample = jax.random.split(rng_input)
        X, y = self.dataloader.sample(rng_sample)
        y_pred = self.network({"params": network_params}, X, rng_net)
        loss, acc = loss_and_acc(y_pred, y, self.num_classes)
        # Return negative loss to maximize!
        return -1 * loss, acc

    @property
    def input_shape(self) -> Tuple[int]:
        """Get the shape of the observation."""
        return (1,) + self.dataloader.data_shape


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array, num_classes: int
) -> Tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


class BatchLoader:
    def __init__(
        self,
        X: chex.Array,
        y: chex.Array,
        batch_size: int,
    ):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
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
            "to use the `VisionFitness` module."
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
        datasets.MNIST(
            "~/data", download=True, train=not test, transform=transform
        ),
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
            "to use the `VisionFitness` module."
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
            "to use the `VisionFitness` module."
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
            "to use the `VisionFitness` module."
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
