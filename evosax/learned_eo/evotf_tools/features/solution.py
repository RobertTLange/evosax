import functools
from typing import Tuple
import chex
from flax import struct
import jax
import jax.numpy as jnp
from evosax.core.fitness import range_norm_trafo
from evosax.utils.helpers import get_best_fitness_member


@struct.dataclass
class SolutionFeaturesState:
    best_fitness: float
    best_member: chex.Array
    gen_counter: int


class SolutionFeaturizer(object):
    def __init__(
        self,
        popsize: int,
        num_dims: int,
        seq_len: int,
        norm_diff_mean: bool = False,
        norm_diff_mean_sq: bool = False,
        diff_best: bool = False,
        norm_diff_best: bool = False,
        maximize: bool = False,
        verbose: bool = False,
    ):
        self.popsize = popsize
        self.num_dims = num_dims
        self.seq_len = seq_len
        self.norm_diff_mean = norm_diff_mean
        self.norm_diff_mean_sq = norm_diff_mean_sq
        self.diff_best = diff_best
        self.norm_diff_best = norm_diff_best
        self.maximize = maximize
        self.verbose = verbose

        if self.verbose:
            print(
                f"Solution Features / Batch shape: {self.num_features} / {self.example_batch_shape}"
            )
            print("[BASE] Diff mean -> (x - mu)")
            print(f"[{self.norm_diff_mean}] Norm difference mean -> (x - mu)/ sigma")
            print(
                f"[{self.norm_diff_mean_sq}] Norm difference to mean squared -> [(x - mu)/ sigma]^2"
            )
            print(f"[{self.diff_best}] Difference to best -> (x - x_best)")
            print(
                f"[{self.norm_diff_best}] Normalized difference to best  -> (x - x_best) in [-0.5, 0.5]"
            )

    @functools.partial(jax.jit, static_argnums=0)
    def initialize(self):
        return SolutionFeaturesState(
            best_fitness=jnp.finfo(jnp.float32).max,
            best_member=jnp.zeros((self.num_dims,)),
            gen_counter=0,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def featurize(
        self,
        x: chex.Array,
        fitness: chex.Array,
        mean: chex.Array,
        sigma: chex.Array,
        state: SolutionFeaturesState,
    ) -> chex.Array:
        diff_mean = x - mean
        sol_out = jnp.expand_dims(x, axis=-1)

        if self.norm_diff_mean:
            norm_diff_mean = jnp.expand_dims(diff_mean / sigma, axis=-1)
            sol_out = jnp.concatenate([sol_out, norm_diff_mean], axis=-1)

        if self.norm_diff_mean_sq:
            norm_diff_mean_sq = jnp.expand_dims(jnp.square(diff_mean / sigma), axis=-1)
            sol_out = jnp.concatenate([sol_out, norm_diff_mean_sq], axis=-1)

        best_member, best_fitness = get_best_fitness_member(
            x, fitness, state, self.maximize
        )
        if self.diff_best:
            diff_best = jnp.expand_dims(x - best_member, axis=-1)
            sol_out = jnp.concatenate([sol_out, diff_best], axis=-1)

        if self.norm_diff_best:
            dist_best_norm = jax.vmap(
                range_norm_trafo, in_axes=(1, None, None), out_axes=1
            )(x - best_member, -0.5, 0.5)
            dist_best_norm = jnp.expand_dims(dist_best_norm, axis=-1)
            sol_out = jnp.concatenate([sol_out, dist_best_norm], axis=-1)
        return sol_out, state.replace(
            best_member=best_member,
            best_fitness=best_fitness,
            gen_counter=state.gen_counter + 1,
        )

    @property
    def num_features(self) -> int:
        return (
            1
            + self.norm_diff_mean
            + self.norm_diff_mean_sq
            + self.diff_best
            + self.norm_diff_best
        )

    @property
    def example_batch_shape(self) -> Tuple[int, ...]:
        return (
            1,  # Batchsize
            self.seq_len,  # Timesteps
            self.popsize,  # Popsize
            self.num_dims,  # Number of dims
            self.num_features,  # Solution features
        )
