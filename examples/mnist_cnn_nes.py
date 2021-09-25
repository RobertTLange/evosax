import optax
import haiku as hk

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp
from evosax.strategies.open_nes import init_strategy, ask, tell
from evosax.utils import (rank_shaped_fitness, get_total_params,
                          get_network_shapes, flat_to_network)
import tensorflow_datasets as tfds


def load_dataset(split, is_training, batch_size):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


class MNIST_CNN(hk.Module):
    def __init__(self, output_channels=(8, 16, 16), strides=(2, 1, 2)):
        super().__init__()
        self.output_channels = output_channels
        self.strides = strides

    def __call__(self, batch):
        """Classifies images as real or fake."""
        x = batch["image"].astype(jnp.float32) / 255.
        for output_channels, stride in zip(self.output_channels, self.strides):
            x = hk.Conv2D(output_channels=output_channels,
                        kernel_shape=[5, 5],
                        stride=stride,
                        padding="SAME")(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = hk.Flatten()(x)
        # We have two classes: 0 = input is fake, 1 = input is real.
        logits = hk.Linear(10)(x)
        return logits

# Evaluation metrics (classification accuracy + cross-entropy loss).
@jax.jit
def accuracy(params, batch):
    predictions = net.apply(params, batch)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])


@jax.jit
def loss(params, batch, w_decay=1e-4):
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    labels = jax.nn.one_hot(batch["label"], 10)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + w_decay * l2_loss


def reshape_and_eval(x, network_shapes):
    """ Perform both parameter reshaping and evaluation in one go. """
    nn = flat_to_network(x, network_shapes)
    out = loss(nn, next(train))
    return out

batch_fitness = vmap(reshape_and_eval, in_axes=(0, None, None))


if __name__ == "__main__":
    # Make MNIST datasets.
    train = load_dataset("train", is_training=True, batch_size=128)
    train_eval = load_dataset("train", is_training=False, batch_size=10000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    # Initialize transform and network size
    rng = jax.random.PRNGKey(0)
    net = hk.without_apply_rng(hk.transform(lambda *args: MNIST_CNN()(*args)))
    params = net.init(jax.random.PRNGKey(42), next(train))

    # Setup NES
    total_no_params = get_total_params(params)
    network_shapes = get_network_shapes(params)
    pop_size = 32
    num_generations = 2000
    lrate = 1e-4
    sigma_init = 0.1

    mean_init = jnp.zeros((total_no_params,))
    opt = optax.adam(lrate)
    opt_state = opt.init(mean_init)

    params, memory = init_strategy(lrate, mean_init, sigma_init,
                                   population_size)
    fit = []
    for g in range(num_generations):
        # Explicitly handle random number generation
        rng, rng_input = jax.random.split(rng)

        # Ask for the next generation population to test
        x, memory = ask(rng_input, params, memory)
        # Evaluate the fitness of the generation members
        #fitness = batch_rosenbrock(x, 1, 100)
        fitness = batch_fitness(x, network_shapes)
        fit.append(fitness.min())
        fitness = z_score_fitness(fitness)
        # x, fitness = rank_shaped_fitness(x, fitness)
        # Tell/Update the CMA-ES with newest data points
        memory, opt_state = tell(x, fitness, params, memory, opt_state)
