import functools

import jax
import jax.numpy as jnp
from flax import struct

from evosax.algorithms.base import update_best_solution_and_fitness
from evosax.core.fitness_shaping import range_norm_trafo


@struct.dataclass
class SolutionFeaturesState:
    best_fitness: float
    best_solution: jax.Array
    generation_counter: int


class SolutionFeaturizer:
    def __init__(
        self,
        population_size: int,
        num_dims: int,
        seq_len: int,
        norm_diff_mean: bool = False,
        norm_diff_mean_sq: bool = False,
        diff_best: bool = False,
        norm_diff_best: bool = False,
        maximize: bool = False,
        verbose: bool = False,
    ):
        self.population_size = population_size
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
            print(f"[{self.norm_diff_mean}] Norm difference mean -> (x - mu)/ std")
            print(
                f"[{self.norm_diff_mean_sq}] Norm difference to mean squared -> [(x - mu)/ std]^2"
            )
            print(f"[{self.diff_best}] Difference to best -> (x - x_best)")
            print(
                f"[{self.norm_diff_best}] Normalized difference to best  -> (x - x_best) in [-0.5, 0.5]"
            )

    @functools.partial(jax.jit, static_argnames=("self",))
    def init(self):
        return SolutionFeaturesState(
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    @functools.partial(jax.jit, static_argnames=("self",))
    def featurize(
        self,
        population: jax.Array,
        fitness: jax.Array,
        mean: jax.Array,
        std: jax.Array,
        state: SolutionFeaturesState,
    ) -> jax.Array:
        diff_mean = population - mean
        sol_out = jnp.expand_dims(population, axis=-1)

        if self.norm_diff_mean:
            norm_diff_mean = jnp.expand_dims(diff_mean / std, axis=-1)
            sol_out = jnp.concatenate([sol_out, norm_diff_mean], axis=-1)

        if self.norm_diff_mean_sq:
            norm_diff_mean_sq = jnp.expand_dims(jnp.square(diff_mean / std), axis=-1)
            sol_out = jnp.concatenate([sol_out, norm_diff_mean_sq], axis=-1)

        if self.maximize:
            best_solution, best_fitness = update_best_solution_and_fitness(
                population, -fitness, state.best_solution, -state.best_fitness
            )
            best_fitness = -best_fitness
        else:
            best_solution, best_fitness = update_best_solution_and_fitness(
                population, fitness, state.best_solution, state.best_fitness
            )
        if self.diff_best:
            diff_best = jnp.expand_dims(population - best_solution, axis=-1)
            sol_out = jnp.concatenate([sol_out, diff_best], axis=-1)

        if self.norm_diff_best:
            dist_best_norm = jax.vmap(
                range_norm_trafo, in_axes=(1, None, None), out_axes=1
            )(population - best_solution, -0.5, 0.5)
            dist_best_norm = jnp.expand_dims(dist_best_norm, axis=-1)
            sol_out = jnp.concatenate([sol_out, dist_best_norm], axis=-1)
        return sol_out, state.replace(
            best_solution=best_solution,
            best_fitness=best_fitness,
            generation_counter=state.generation_counter + 1,
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
    def example_batch_shape(self) -> tuple[int, ...]:
        return (
            1,
            self.seq_len,
            self.population_size,
            self.num_dims,
            self.num_features,
        )
