import functools

import jax
import jax.numpy as jnp
from flax import struct

from evosax.algorithms.distribution_based.xnes import get_weights as get_nes_weights
from evosax.core.fitness_shaping import centered_rank_trafo


class TraceConstructor:
    def __init__(self, num_dims: int, timescales: jax.Array):
        self.num_dims = num_dims
        self.timescales = timescales

    def init(self) -> jax.Array:
        return jnp.zeros((self.num_dims, self.timescales.shape[0]))

    def update(self, paths: jax.Array, diff: jax.Array) -> jax.Array:
        def update_path(lrate, path, diff):
            return (1 - lrate) * path + lrate * diff

        return jax.vmap(update_path, in_axes=(0, 1, None), out_axes=1)(
            self.timescales, paths, diff
        )


@struct.dataclass
class DistributionFeaturesState:
    old_mean: jax.Array
    old_std: jax.Array
    momentum_mean: jax.Array
    momentum_std: jax.Array
    evopath_mean: jax.Array
    evopath_std: jax.Array


class DistributionFeaturizer:
    def __init__(
        self,
        population_size: int,
        num_dims: int,
        seq_len: int,
        use_mean: bool = False,
        use_sigma: bool = False,
        use_evopaths: bool = False,
        use_momentum: bool = False,
        evopath_timescales: list[float] = [0.9, 0.95, 1.0],
        momentum_timescales: list[float] = [0.9, 0.95, 1.0],
        use_oai_grad: bool = False,
        verbose: bool = False,
    ):
        self.population_size = population_size
        self.num_dims = num_dims
        self.seq_len = seq_len
        self.use_mean = use_mean
        self.use_std = use_sigma
        self.use_evopaths = use_evopaths
        self.use_momentum = use_momentum
        self.evopath_timescales = evopath_timescales
        self.momentum_timescales = momentum_timescales
        self.use_oai_grad = use_oai_grad
        self.verbose = verbose

        self.evopath = TraceConstructor(num_dims, jnp.array(evopath_timescales))
        self.mompath = TraceConstructor(num_dims, jnp.array(momentum_timescales))

        if self.verbose:
            print(
                f"Distribution Features / Batch shape: {self.num_features} / {self.example_batch_shape}"
            )
            print("[BASE] SNES grad mean -> sum w_i (x-mu)/std")
            print("[BASE] SNES grad std -> sum w_i [(x-mu)/std]**2 - 1")
            print(f"[{self.use_mean}] Mean")
            print(f"[{self.use_std}] std")
            print(f"[{self.use_evopaths}] Evolution Paths -> {self.evopath_timescales}")
            print(f"[{self.use_momentum}] Momentum -> {self.momentum_timescales}")
            print(f"[{self.use_oai_grad}] OpenAI Gradient")

    @functools.partial(jax.jit, static_argnames=("self",))
    def init(self) -> DistributionFeaturesState:
        return DistributionFeaturesState(
            old_mean=jnp.zeros((self.num_dims,)),
            old_std=jnp.ones((self.num_dims,)),
            momentum_mean=self.mompath.init(),
            momentum_std=self.mompath.init(),
            evopath_mean=self.evopath.init(),
            evopath_std=self.evopath.init(),
        )

    @functools.partial(jax.jit, static_argnames=("self",))
    def featurize(
        self,
        x: jax.Array,
        fitness: jax.Array,
        mean: jax.Array,
        std: jax.Array,
        state: DistributionFeaturesState,
    ) -> jax.Array:
        ranks = fitness.argsort()
        weights = get_nes_weights(fitness.shape[0])
        noise = (x - mean) / std
        sorted_noise = noise[ranks]
        grad_mean = jnp.dot(weights, sorted_noise).reshape(-1, 1)
        grad_std = jnp.dot(weights, sorted_noise**2 - 1).reshape(-1, 1)
        distrib_features = jnp.concatenate([grad_mean, grad_std], axis=1)
        if self.use_mean:
            distrib_features = jnp.concatenate(
                [distrib_features, mean.reshape(-1, 1)], axis=1
            )
        if self.use_std:
            distrib_features = jnp.concatenate(
                [distrib_features, std.reshape(-1, 1)], axis=1
            )

        # Update traces
        path_mean = self.evopath.update(state.evopath_mean, grad_mean.squeeze())
        path_std = self.evopath.update(state.evopath_std, grad_std.squeeze())
        if self.use_evopaths:
            distrib_features = jnp.concatenate(
                [distrib_features, path_mean, path_std], axis=1
            )
        mom_mean = self.mompath.update(state.momentum_mean, mean - state.old_mean)
        mom_std = self.mompath.update(state.momentum_std, std - state.old_std)
        if self.use_momentum:
            distrib_features = jnp.concatenate(
                [distrib_features, mom_mean, mom_std], axis=1
            )
        if self.use_oai_grad:
            population_size = x.shape[0]
            fitness_re = centered_rank_trafo(fitness)
            oai_grad = 1.0 / (population_size * std) * jnp.dot(noise.T, fitness_re)
            distrib_features = jnp.concatenate(
                [distrib_features, oai_grad.reshape(-1, 1)], axis=1
            )
        return distrib_features, DistributionFeaturesState(
            mean, std, mom_mean, mom_std, path_mean, path_std
        )

    @property
    def num_features(self):
        return (
            2  # SNES grad mean, std
            + 2 * len(self.evopath_timescales) * self.use_evopaths  # evopaths
            + 2 * len(self.momentum_timescales) * self.use_momentum  # momentum
            + self.use_mean  # mean
            + self.use_std  # std
            + self.use_oai_grad  # oai grad
        )

    @property
    def example_batch_shape(self):
        return (
            1,  # Batchsize
            self.seq_len,  # Timesteps
            self.num_dims,  # Number of dims
            self.num_features,  # Distribution features
        )
