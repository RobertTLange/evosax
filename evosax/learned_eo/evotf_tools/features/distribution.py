import functools
from typing import List
import chex
from flax import struct
import jax
import jax.numpy as jnp
from evosax.strategies.snes import get_snes_weights
from evosax.core.fitness import centered_rank_trafo


class TraceConstructor(object):
    def __init__(self, num_dims: int, timescales: chex.Array):
        self.num_dims = num_dims
        self.timescales = timescales

    def initialize(self) -> chex.Array:
        return jnp.zeros((self.num_dims, self.timescales.shape[0]))

    def update(self, paths: chex.Array, diff: chex.Array) -> chex.Array:
        def update_path(lrate, path, diff):
            return (1 - lrate) * path + lrate * diff

        return jax.vmap(update_path, in_axes=(0, 1, None), out_axes=1)(
            self.timescales, paths, diff
        )


@struct.dataclass
class DistributionFeaturesState:
    old_mean: chex.Array
    old_sigma: chex.Array
    momentum_mean: chex.Array
    momentum_sigma: chex.Array
    evopath_mean: chex.Array
    evopath_sigma: chex.Array


class DistributionFeaturizer(object):
    def __init__(
        self,
        popsize: int,
        num_dims: int,
        seq_len: int,
        use_mean: bool = False,
        use_sigma: bool = False,
        use_evopaths: bool = False,
        use_momentum: bool = False,
        evopath_timescales: List[float] = [0.9, 0.95, 1.0],
        momentum_timescales: List[float] = [0.9, 0.95, 1.0],
        use_oai_grad: bool = False,
        verbose: bool = False,
    ):
        self.popsize = popsize
        self.num_dims = num_dims
        self.seq_len = seq_len
        self.use_mean = use_mean
        self.use_sigma = use_sigma
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
            print("[BASE] SNES grad mean -> sum w_i (x-mu)/sigma")
            print("[BASE] SNES grad sigma -> sum w_i [(x-mu)/sigma]**2 - 1")
            print(f"[{self.use_mean}] Mean")
            print(f"[{self.use_sigma}] Sigma")
            print(f"[{self.use_evopaths}] Evolution Paths -> {self.evopath_timescales}")
            print(f"[{self.use_momentum}] Momentum -> {self.momentum_timescales}")
            print(f"[{self.use_oai_grad}] OpenAI Gradient")

    @functools.partial(jax.jit, static_argnums=0)
    def initialize(self) -> DistributionFeaturesState:
        return DistributionFeaturesState(
            old_mean=jnp.zeros((self.num_dims,)),
            old_sigma=jnp.ones((self.num_dims,)),
            momentum_mean=self.mompath.initialize(),
            momentum_sigma=self.mompath.initialize(),
            evopath_mean=self.evopath.initialize(),
            evopath_sigma=self.evopath.initialize(),
        )

    @functools.partial(jax.jit, static_argnums=0)
    def featurize(
        self,
        x: chex.Array,
        fitness: chex.Array,
        mean: chex.Array,
        sigma: chex.Array,
        state: DistributionFeaturesState,
    ) -> chex.Array:
        ranks = fitness.argsort()
        weights = get_snes_weights(fitness.shape[0])
        noise = (x - mean) / sigma
        sorted_noise = noise[ranks]
        grad_mean = (weights * sorted_noise).sum(axis=0).reshape(-1, 1)
        grad_sigma = (weights * (sorted_noise**2 - 1)).sum(axis=0).reshape(-1, 1)
        distrib_features = jnp.concatenate([grad_mean, grad_sigma], axis=1)
        if self.use_mean:
            distrib_features = jnp.concatenate(
                [distrib_features, mean.reshape(-1, 1)], axis=1
            )
        if self.use_sigma:
            distrib_features = jnp.concatenate(
                [distrib_features, sigma.reshape(-1, 1)], axis=1
            )

        # Update traces
        path_mean = self.evopath.update(state.evopath_mean, grad_mean.squeeze())
        path_sigma = self.evopath.update(state.evopath_sigma, grad_sigma.squeeze())
        if self.use_evopaths:
            distrib_features = jnp.concatenate(
                [distrib_features, path_mean, path_sigma], axis=1
            )
        mom_mean = self.mompath.update(state.momentum_mean, mean - state.old_mean)
        mom_sigma = self.mompath.update(state.momentum_sigma, sigma - state.old_sigma)
        if self.use_momentum:
            distrib_features = jnp.concatenate(
                [distrib_features, mom_mean, mom_sigma], axis=1
            )
        if self.use_oai_grad:
            popsize = x.shape[0]
            fitness_re = centered_rank_trafo(fitness)
            oai_grad = 1.0 / (popsize * sigma) * jnp.dot(noise.T, fitness_re)
            distrib_features = jnp.concatenate(
                [distrib_features, oai_grad.reshape(-1, 1)], axis=1
            )
        return distrib_features, DistributionFeaturesState(
            mean, sigma, mom_mean, mom_sigma, path_mean, path_sigma
        )

    @property
    def num_features(self):
        return (
            2  # SNES grad mean, sigma
            + 2 * len(self.evopath_timescales) * self.use_evopaths  # evopaths
            + 2 * len(self.momentum_timescales) * self.use_momentum  # momentum
            + self.use_mean  # mean
            + self.use_sigma  # sigma
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
