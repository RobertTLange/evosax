import functools
import io
import pickle
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .fitness_shaping import (
    centered_rank,
    l2_norm_sq,
    normalize,
    standardize,
)


def _restore_array(data: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Restore a checkpoint array using NumPy's public API."""
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()


def _restore_legacy_jax_array(
    reconstruct: Callable[..., np.ndarray],
    args: tuple[Any, ...],
    array_state: tuple[Any, ...],
    aval_state: dict[str, Any],
) -> np.ndarray:
    """Restore an old pickled JAX array as a portable NumPy array."""
    del aval_state
    value = reconstruct(*args)
    value.__setstate__(array_state)
    return value


def _restore_checkpoint_array(
    current_reconstruct: Callable[..., Any] | None,
    reconstruct: Callable[..., np.ndarray],
    args: tuple[Any, ...],
    array_state: tuple[Any, ...],
    aval_state: dict[str, Any],
) -> Any:
    if current_reconstruct is not None and "named_shape" not in aval_state:
        return current_reconstruct(reconstruct, args, array_state, aval_state)
    return _restore_legacy_jax_array(reconstruct, args, array_state, aval_state)


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            try:
                current_reconstruct = super().find_class(module, name)
            except (AttributeError, ImportError):
                current_reconstruct = None
            return functools.partial(_restore_checkpoint_array, current_reconstruct)
        return super().find_class(module, name)


def load_pkl_object(filename: Any, pkg_load: bool = False) -> Any:
    """Load a trusted checkpoint, including arrays pickled by older JAX releases.

    Pickle loading can execute arbitrary code. Only load checkpoints from trusted
    sources.
    """
    if not pkg_load:
        with open(filename, "rb") as input:
            return _CheckpointUnpickler(input).load()
    return _CheckpointUnpickler(io.BytesIO(filename)).load()


def device_put_parameter_arrays(tree: Any) -> Any:
    """Transfer NumPy parameter leaves to the default JAX device once."""
    return jax.tree.map(
        lambda leaf: (
            jax.device_put(leaf) if isinstance(leaf, (np.ndarray, np.generic)) else leaf
        ),
        tree,
    )


def tanh_timestamp(x: int | jax.Array) -> jax.Array:
    """Timestamp embedding with evo-adapted timescales (Metz et al., 2022)."""

    def single_frequency(timescale):
        return jnp.tanh(x / jnp.float32(timescale) - 1.0)

    all_frequencies = jnp.asarray(
        [1, 3, 10, 30, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        dtype=jnp.float32,
    )
    return jax.vmap(single_frequency)(all_frequencies)


class EvolutionPath:
    def __init__(self, num_dims: int, timescales: jax.Array):
        self.num_dims = num_dims
        self.timescales = timescales

    def init(self) -> jax.Array:
        """Initialize evolution path arrays."""
        return jnp.zeros((self.num_dims, self.timescales.shape[0]))

    def update(self, paths: jax.Array, diff: jax.Array) -> jax.Array:
        """Batch update evolution paths for multiple dims & timescales."""

        def update_path(lr, path, diff):
            return (1 - lr) * path + lr * diff

        return jax.vmap(update_path, in_axes=(0, 1, None), out_axes=1)(
            self.timescales, paths, diff
        )


class FitnessFeatures:
    """Fitness Feature Constructor."""

    def __init__(
        self,
        centered_rank: bool = False,
        z_score: bool = False,
        w_decay: float = 0.0,
        diff_best: bool = False,
        norm_range: bool = False,
        maximize: bool = False,
    ):
        self.centered_rank = centered_rank
        self.z_score = z_score
        self.w_decay = w_decay
        self.diff_best = diff_best
        self.norm_range = norm_range
        self.maximize = maximize

    @functools.partial(jax.jit, static_argnames=("self",))
    def apply(self, x: jax.Array, fitness: jax.Array, best_fitness: float) -> jax.Array:
        """Compute and concatenate different fitness transformations."""
        fitness = jax.lax.select(self.maximize, -1 * fitness, fitness)
        fit_out = ((fitness < best_fitness) * 1.0).reshape(-1, 1)

        if self.centered_rank:
            fit_cr = centered_rank(fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_cr], axis=1)
        if self.z_score:
            fit_zs = standardize(fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_zs], axis=1)
        if self.diff_best:
            fit_best = norm_diff_best(fitness, best_fitness).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_best], axis=1)
        if self.norm_range:
            fit_norm = normalize(fitness, -1.0, 1.0).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_norm], axis=1)
        if self.w_decay:
            fit_wnorm = l2_norm_sq(x).reshape(-1, 1)
            fit_out = jnp.concatenate([fit_out, fit_wnorm], axis=1)
        return fit_out


def norm_diff_best(fitness: jax.Array, best_fitness: float) -> jax.Array:
    """Normalize difference from best previous fitness score."""
    fitness = jnp.clip(fitness, -1e10, 1e10)
    diff_best = fitness - best_fitness
    return jnp.clip(
        diff_best / (jnp.nanmax(diff_best) - jnp.nanmin(diff_best) + 1e-10),
        -1,
        1,
    )


class AttentionWeights(nn.Module):
    """Self-attention layer for recombination weights."""

    att_hidden_dims: int = 8

    @nn.compact
    def __call__(self, X: jax.Array) -> jax.Array:
        keys = nn.Dense(self.att_hidden_dims)(X)
        queries = nn.Dense(self.att_hidden_dims)(X)
        values = nn.Dense(1)(X)
        A = nn.softmax(jnp.matmul(queries, keys.T) / jnp.sqrt(X.shape[0]))
        weights = nn.softmax(jnp.matmul(A, values).squeeze())
        return weights[:, None]


class EvoPathMLP(nn.Module):
    """MLP layer for learning rate modulation based on evopaths."""

    mlp_hidden_dims: int = 8

    @nn.compact
    def __call__(
        self,
        path_c: jax.Array,
        path_sigma: jax.Array,
        time_embed: jax.Array,
    ):
        timestamps = jnp.repeat(time_embed[None, ...], repeats=path_c.shape[0], axis=0)
        X = jnp.concatenate([path_c, path_sigma, timestamps], axis=1)
        # Perform MLP hidden state update for each solution dim. in parallel
        hidden = jax.vmap(nn.Dense(self.mlp_hidden_dims), in_axes=(0))(X)
        hidden = nn.relu(hidden)
        lrs_mean = nn.sigmoid(nn.Dense(1)(hidden)).squeeze()
        lrs_sigma = nn.sigmoid(nn.Dense(1)(hidden)).squeeze()
        return lrs_mean, lrs_sigma
