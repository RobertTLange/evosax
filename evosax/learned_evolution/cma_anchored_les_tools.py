"""Dimensionless residual networks and frozen parameters for CMAAnchoredLES."""

import math
import pickle
from importlib.resources import files
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, serialization
from flax.traverse_util import flatten_dict

from evosax.types import PyTree

from .les_tools import load_pkl_object

LN2 = math.log(2.0)
HIDDEN_DIMS = 8
MEMBER_FEATURES = 6
GLOBAL_FEATURES = 9
DIMENSION_FEATURES = 6
INIT_SET_SIZE = 4
SUCCESS_RATES = jnp.asarray([0.05, 0.2, 0.6])
C_MEAN = 1.0
INNER_STD_INIT = 0.05
RESTART_EXPANSION = 3.0
ANTITHETIC_SAMPLING = True
PATH_PAIR_CORRECTION = True
IMPROVER_BUDGETED_MIRRORING = True
GATE_GAIN = 3.0
LOG_SIGMA_STEP_BOUND = 1.0
LOG_AXIS_STEP_BOUND = 1.0
LOG_AXIS_BOUND = 15.0
SIGMA_MIN_FACTOR = 1e-09
SIGMA_MAX_FACTOR = 100000.0
EPS = 1e-08


def fitness_ranks(fitness: jax.Array) -> jax.Array:
    """Return the ascending rank of each member (0 = best) of a minimized fitness."""
    indices = jnp.argsort(fitness)
    return indices.at[indices].set(jnp.arange(fitness.size))


def bounded_gate(value: jax.Array) -> jax.Array:
    """Map an unbounded network output to a multiplicative gate in ``[1/2, 2]``."""
    return jnp.exp(LN2 * jnp.tanh(GATE_GAIN * value))


def cma_constants(num_dims: int, population_size: int) -> dict[str, Any]:
    """Derive the sep-CMA-ES constants from generic ``D`` and ``N`` formulas."""
    dims = float(num_dims)
    num_elites = max(1, population_size // 2)
    raw_weights = [
        math.log(num_elites + 0.5) - math.log(index + 1.0)
        for index in range(num_elites)
    ]
    total = sum(raw_weights)
    positive_weights = [weight / total for weight in raw_weights]
    mu_eff = 1.0 / sum(weight * weight for weight in positive_weights)
    c_sigma = (mu_eff + 2.0) / (dims + mu_eff + 5.0)
    d_sigma = (
        1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dims + 1.0)) - 1.0) + c_sigma
    )
    c_c = (4.0 + mu_eff / dims) / (dims + 4.0 + 2.0 * mu_eff / dims)
    alpha_cov = (dims + 2.0) / 3.0
    c_1 = alpha_cov * 2.0 / ((dims + 1.3) ** 2 + mu_eff)
    c_mu = (
        alpha_cov
        * 2.0
        * max(0.0, mu_eff - 2.0 + 1.0 / mu_eff)
        / ((dims + 2.0) ** 2 + mu_eff)
    )
    if c_1 + c_mu > 1.0:
        rescale = 1.0 / (c_1 + c_mu)
        c_1 *= rescale
        c_mu *= rescale
    chi_d = math.sqrt(dims) * (1.0 - 1.0 / (4.0 * dims) + 1.0 / (21.0 * dims * dims))
    base_weights = jnp.asarray(
        positive_weights + [0.0] * (population_size - num_elites), dtype=jnp.float32
    )
    return {
        "base_weights": base_weights,
        "c_sigma": c_sigma,
        "d_sigma": d_sigma,
        "c_c": c_c,
        "c_1": c_1,
        "c_mu": c_mu,
        "chi_d": chi_d,
    }


class RecombinationResidual(nn.Module):
    """Permutation-equivariant set module emitting bounded weight residuals."""

    hidden_dims: int = HIDDEN_DIMS

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        hidden = nn.relu(nn.Dense(self.hidden_dims)(features))
        context = jnp.mean(hidden, axis=0, keepdims=True)
        joined = jnp.concatenate(
            (hidden, jnp.broadcast_to(context, hidden.shape)), axis=-1
        )
        hidden = nn.relu(nn.Dense(self.hidden_dims)(joined))
        residual = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )(hidden)
        return residual[..., 0]


class GlobalGates(nn.Module):
    """Scalar step-size and mean gates conditioned on global search evidence."""

    hidden_dims: int = HIDDEN_DIMS

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        hidden = nn.relu(nn.Dense(self.hidden_dims)(features))
        return nn.Dense(
            2, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )(hidden)


class DimensionGate(nn.Module):
    """Per-dimension covariance gate, shared across dimensions."""

    hidden_dims: int = HIDDEN_DIMS

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        hidden = nn.relu(nn.Dense(self.hidden_dims)(features))
        gate = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros
        )(hidden)
        return gate[..., 0]


def initialize_meta_parameters(key: jax.Array) -> PyTree:
    """Initialize every learned weight from fixed-size, D/N-independent dummies."""
    (recombination_key, global_key, dimension_key) = jax.random.split(key, 3)
    return {
        "recombination": RecombinationResidual().init(
            recombination_key, jnp.zeros((INIT_SET_SIZE, MEMBER_FEATURES))
        ),
        "global_gates": GlobalGates().init(global_key, jnp.zeros((GLOBAL_FEATURES,))),
        "dimension_gate": DimensionGate().init(
            dimension_key, jnp.zeros((INIT_SET_SIZE, DIMENSION_FEATURES))
        ),
    }


def load_parameters(params: PyTree | None, params_path: str | None) -> PyTree:
    """Load trusted pickled weights and validate the fixed network shapes.

    Pickle loading can execute code. Only use checkpoints from trusted sources.
    """
    if params is None:
        path = (
            Path(params_path)
            if params_path is not None
            else files("evosax.algorithms").joinpath(
                "ckpt/cma_anchored_les/2026_10_cma_anchored_les.pkl"
            )
        )
        try:
            params = load_pkl_object(path.read_bytes(), pkg_load=True)
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid CMAAnchoredLES checkpoint: {path}") from error
    template = initialize_meta_parameters(jax.random.key(0))
    try:
        candidate = serialization.to_state_dict(params)
        expected = flatten_dict(template)
        actual = flatten_dict(candidate)
        if actual.keys() != expected.keys():
            raise ValueError("network structure differs")
        for name, value in actual.items():
            array = np.asarray(value)
            if array.shape != expected[name].shape:
                raise ValueError(f"wrong shape at {'/'.join(name)}")
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
                raise ValueError(f"non-real numeric array at {'/'.join(name)}")
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite array at {'/'.join(name)}")
        result = jax.tree.map(jnp.asarray, candidate)
        if not all(np.isfinite(value).all() for value in jax.tree.leaves(result)):
            raise ValueError("array overflow during device conversion")
        return result
    except (AssertionError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CMAAnchoredLES parameters: {error}") from error
