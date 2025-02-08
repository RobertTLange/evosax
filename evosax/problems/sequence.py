from functools import partial

import chex
import jax
import jax.numpy as jnp


class SequenceFitness:
    def __init__(
        self,
        task_name: str = "SeqMNIST",
        batch_size: int = 128,
        seq_length: int = 150,  # Sequence length in addition task
        permute_seq: bool = False,  # Permuted S-MNIST task option
        test: bool = False,
    ):
        self.task_name = task_name
        self.batch_size = batch_size
        self.steps_per_member = 1
        self.test = test

        # Setup task-specific input/output shapes and loss fn
        if self.task_name == "SeqMNIST":
            self.action_shape = 10
            self.permute_seq = permute_seq
            self.seq_length = 784
            self.loss_fn = partial(loss_and_acc, num_classes=10)
        elif self.task_name == "Addition":
            self.action_shape = 1
            self.permute_seq = False
            self.seq_length = seq_length
            self.loss_fn = loss_and_mae
        else:
            raise ValueError("Dataset is not supported.")

        data = get_array_data(
            self.task_name, self.seq_length, self.permute_seq, self.test
        )
        self.dataloader = BatchLoader(*data, batch_size=self.batch_size)
        self.num_rnn_steps = self.dataloader.data_shape[1]

    def set_apply_fn(self, network, carry_init):
        """Set the network forward function."""
        self.network = network
        self.carry_init = carry_init
        self.rollout_pop = jax.vmap(self.rollout_rnn, in_axes=(None, 0))
        self.rollout = jax.jit(self.rollout_vmap)

    def rollout_vmap(self, key: jax.Array, network_params: chex.ArrayTree):
        """Vectorize rollout. Reshape output correctly."""
        loss, perf = self.rollout_pop(key, network_params)
        loss_re = loss.reshape(-1, 1)
        perf_re = perf.reshape(-1, 1)
        return loss_re, perf_re

    def rollout_rnn(
        self, key: jax.Array, network_params: chex.ArrayTree
    ) -> tuple[float, float]:
        """Evaluate a network on a supervised learning task."""
        key_sample, key_rollout = jax.random.split(key)
        X, y = self.dataloader.sample(key_sample)
        # Map over sequence batch dimension
        y_pred = jax.vmap(self.rollout_single, in_axes=(None, None, 0))(
            key_rollout, network_params, X
        )
        loss, perf = self.loss_fn(y_pred, y)
        # Return negative loss to maximize!
        return -1 * loss, perf

    def rollout_single(
        self,
        key: jax.Array,
        network_params: chex.ArrayTree,
        X_single: chex.ArrayTree,
    ):
        """Rollout RNN on a single sequence."""
        # Reset the network
        hidden = self.carry_init()

        def rnn_step(carry, _):
            """lax.scan compatible step transition in jax env."""
            network_params, hidden, key, t = carry
            key, key_network = jax.random.split(key)
            hidden, pred = self.network(
                network_params,
                X_single[t],
                hidden,
                key_network,
            )
            return (network_params, hidden, key, t + 1), pred

        # Scan over image length (784)/sequence
        _, scan_out = jax.lax.scan(
            rnn_step,
            (network_params, hidden, key, 0),
            length=self.num_rnn_steps,
        )
        y_pred = scan_out[-1]
        return y_pred

    @property
    def input_shape(self) -> tuple[int]:
        """Get the shape of the observation."""
        return self.dataloader.data_shape


def loss_and_acc(
    y_pred: chex.Array, y_true: chex.Array, num_classes: int
) -> tuple[chex.Array, chex.Array]:
    """Compute cross-entropy loss and accuracy."""
    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
    labels = jax.nn.one_hot(y_true, num_classes)
    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
    loss /= labels.shape[0]
    return loss, acc


def loss_and_mae(
    y_pred: chex.Array, y_true: chex.Array
) -> tuple[chex.Array, chex.Array]:
    """Compute mean squared error loss and mean absolute error."""
    loss = jnp.mean((y_pred.squeeze() - y_true) ** 2)
    mae = jnp.mean(jnp.abs(y_pred.squeeze() - y_true))
    return loss, -mae


class BatchLoader:
    def __init__(
        self,
        X: chex.Array,
        y: chex.Array,
        batch_size: int,
    ):
        self.X = X
        self.y = y
        self.data_shape = self.X.shape[1:][::-1]
        self.num_train_samples = X.shape[0]
        self.batch_size = batch_size

    def sample(self, key: jax.Array) -> tuple[chex.Array, chex.Array]:
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


def get_smnist_loaders(test: bool = False):
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
            transforms.Lambda(lambda x: torch.flatten(x)),
            transforms.Lambda(lambda x: torch.unsqueeze(x, -1)),
        ]
    )
    bs = 10000 if test else 60000
    loader = torch.utils.data.DataLoader(
        datasets.MNIST("~/data", download=True, train=not test, transform=transform),
        batch_size=bs,
        shuffle=False,
    )
    return loader


def get_adding_data(T: int = 150, test: bool = False):
    """Sample a mask, [0, 1] samples and sum of targets for len T.
    Reference:  Martens & Sutskever. ICML, 2011.
    """
    bs = 100000 if test else 10000
    key = jax.random.key(0)
    keys = jax.random.split(key, bs)

    def get_single_addition(key, T):
        key_uniform, key_choice = jax.random.split(key)
        numbers = jax.random.uniform(key_uniform, (T,), minval=0, maxval=1)
        mask_ids = jax.random.choice(key_choice, jnp.arange(T), (2,), replace=False)
        mask = jnp.zeros(T).at[mask_ids].set(1)
        target = jnp.sum(mask * numbers)
        return jnp.stack([numbers, mask], axis=1), target

    batch_seq_gen = jax.vmap(get_single_addition, in_axes=(0, None))
    data, target = batch_seq_gen(keys, T)
    return data, target


def get_array_data(
    task_name: str = "SMNIST",
    seq_length: int = 150,
    permute_seq: bool = False,
    test: bool = False,
):
    """Get raw data arrays to subsample from."""
    if task_name == "SeqMNIST":
        loader = get_smnist_loaders(test)
        for _, (data, target) in enumerate(loader):
            break
        data, target = jnp.array(data), jnp.array(target)

        # Permute the sequence of the pixels if desired.
        if permute_seq:  # bs, T - fix permutation by seed
            key = jax.random.key(0)
            idx = jnp.arange(784)
            idx_perm = jax.random.permutation(key, idx)
            data = data.at[:].set(data[:, idx_perm])
    elif task_name == "Addition":
        data, target = get_adding_data(seq_length, test)
        data, target = jnp.array(data), jnp.array(target)
    else:
        raise ValueError("Dataset is not supported.")
    return data, target
