"""Blackbox Optimization Benchmarking Noise Models.

This module implements the noise models used in the noisy BBOB suite from [1].

Noise models:
- Noiseless
- Gaussian
- Uniform
- Cauchy
- Additive

[1] https://inria.hal.science/inria-00369466
[2] https://numbbo.github.io/temp-doc-bbob/bbob-noisy/def.html
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class NoiseParams:
    noise_id: jax.Array
    gaussian_beta: jax.Array
    uniform_alpha: jax.Array
    uniform_beta: jax.Array
    cauchy_alpha: jax.Array
    cauchy_p: jax.Array
    additive_std: jax.Array


class NoiseModel:
    """Blackbox Optimization Benchmarking Noise Models class."""

    def __init__(
        self,
        noise_model_names: list[str] = [
            "noiseless",
            "gaussian",
            "uniform",
            "cauchy",
            "additive",
        ],
        noise_ranges: dict[str, tuple[float, float]] = {
            "gaussian_beta": None,
            "uniform_alpha": None,
            "uniform_beta": None,
            "cauchy_alpha": None,
            "cauchy_p": None,
            "additive_std": None,
        },
        use_stabilization: bool = False,
    ):
        # Collect active noise models
        self.noise_ids, self.noise_models, counter = [], [], 0
        for noise_model_name, noise_model in all_noise_models.items():
            if noise_model_name in noise_model_names:
                self.noise_ids.append(counter)
                self.noise_models.append(noise_model)
                counter += 1
        self.noise_ids = jnp.array(self.noise_ids)

        # Default ranges for noise model parameters between moderate and severe
        self.noise_ranges = {
            "gaussian_beta": noise_ranges["gaussian_beta"]
            if noise_ranges["gaussian_beta"]
            else (0.01, 1.0),
            "uniform_alpha": noise_ranges["uniform_alpha"]
            if noise_ranges["uniform_alpha"]
            else (0.005, 0.5),
            "uniform_beta": noise_ranges["uniform_beta"]
            if noise_ranges["uniform_beta"]
            else (0.01, 1.0),
            "cauchy_alpha": noise_ranges["cauchy_alpha"]
            if noise_ranges["cauchy_alpha"]
            else (0.01, 1.0),
            "cauchy_p": noise_ranges["cauchy_p"]
            if noise_ranges["cauchy_p"]
            else (0.05, 0.2),
            "additive_std": noise_ranges["additive_std"]
            if noise_ranges["additive_std"]
            else (0.0, 0.1),
        }

        # Use noise stabilization close to optimal value
        self.use_stabilization = use_stabilization

    def sample_params(self, key: jax.Array) -> NoiseParams:
        """Sample a noise model and its parameter settings."""
        (
            key_id,
            key_gaussian,
            key_uniform_1,
            key_uniform_2,
            key_cauchy_1,
            key_cauchy_2,
            key_additive,
        ) = jax.random.split(key, 7)

        noise_id = jax.random.choice(key_id, self.noise_ids)

        # Sample uniformly between moderate and severe divided by 2
        gaussian_beta = jax.random.uniform(
            key_gaussian,
            minval=self.noise_ranges["gaussian_beta"][0],
            maxval=self.noise_ranges["gaussian_beta"][1],
        )

        uniform_alpha = jax.random.uniform(
            key_uniform_1,
            minval=self.noise_ranges["uniform_alpha"][0],
            maxval=self.noise_ranges["uniform_alpha"][1],
        )
        uniform_beta = jax.random.uniform(
            key_uniform_2,
            minval=self.noise_ranges["uniform_beta"][0],
            maxval=self.noise_ranges["uniform_beta"][1],
        )

        cauchy_alpha = jax.random.uniform(
            key_cauchy_1,
            minval=self.noise_ranges["cauchy_alpha"][0],
            maxval=self.noise_ranges["cauchy_alpha"][1],
        )
        cauchy_p = jax.random.uniform(
            key_cauchy_2,
            minval=self.noise_ranges["cauchy_p"][0],
            maxval=self.noise_ranges["cauchy_p"][1],
        )

        additive_std = jax.random.uniform(
            key_additive,
            minval=self.noise_ranges["additive_std"][0],
            maxval=self.noise_ranges["additive_std"][1],
        )

        return NoiseParams(
            noise_id,
            gaussian_beta,
            uniform_alpha,
            uniform_beta,
            cauchy_alpha,
            cauchy_p,
            additive_std,
        )

    def apply(
        self, key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
    ) -> jax.Array:
        """Apply a noise model given its parameter settings."""
        fn_noise = jax.lax.switch(
            noise_params.noise_id,
            self.noise_models,
            key,
            fn_val,
            noise_params,
        )

        if self.use_stabilization:
            fn_noise = stabilize(fn_val, fn_noise)
        return fn_noise


def noiseless_noise(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Apply noiseless noise."""
    return fn_val


def gaussian_noise(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Apply Gaussian noise ([1], Eq. 1)."""
    # Moderate noise: beta = 0.01
    # Severe noise: beta = 1
    return fn_val * jnp.exp(
        noise_params.gaussian_beta * jax.random.normal(key, shape=fn_val.shape)
    )


def uniform_noise(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Apply uniform noise ([1], Eq. 2)."""
    # Moderate noise: alpha = 0.01 * (0.49 + 1/D), beta = 0.01
    # Severe noise: alpha = 0.49 + 1/D, beta = 1.0
    key_1, key_2 = jax.random.split(key)
    f_1 = jnp.power(
        jax.random.uniform(key_1, shape=fn_val.shape), noise_params.uniform_beta
    )
    f_2 = jnp.power(
        1e9 / (fn_val + 1e-8),
        noise_params.uniform_alpha * jax.random.uniform(key_2, shape=fn_val.shape),
    )
    return fn_val * f_1 * jnp.maximum(1.0, f_2)


def cauchy_noise(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Apply Cauchy noise ([1], Eq. 3)."""
    # Moderate noise: alpha = 0.01, p = 0.05
    # Severe noise: alpha = 1, p = 0.2
    key_1, key_2, key_3 = jax.random.split(key, 3)
    indicator = jax.random.uniform(key_1, shape=fn_val.shape) < noise_params.cauchy_p
    cauchy = jax.random.normal(key_2, shape=fn_val.shape) / (
        jnp.abs(jax.random.uniform(key_3, shape=fn_val.shape)) + 1e-8
    )
    return fn_val + noise_params.cauchy_alpha * jnp.maximum(
        0.0, 1000.0 + indicator * cauchy
    )


def additive_noise(
    key: jax.Array, fn_val: jax.Array, noise_params: NoiseParams
) -> jax.Array:
    """Apply additive noisification."""
    # Moderate noise: std = 0.01
    # Severe noise: std = 1
    return fn_val + noise_params.additive_std * jax.random.normal(
        key, shape=fn_val.shape
    )


def stabilize(
    fn_val: jax.Array, fn_noise: jax.Array, target_value: float = 1e-08
) -> jax.Array:
    """Stabilize final function value ([1], Eq. 4)."""
    # Return undisturbed function value if f is smaller than target value
    return (fn_noise + 1.01 * target_value) * (fn_val >= target_value) + fn_val * (
        fn_val < target_value
    )


all_noise_models = {
    "noiseless": noiseless_noise,
    "gaussian": gaussian_noise,
    "uniform": uniform_noise,
    "cauchy": cauchy_noise,
    "additive": additive_noise,
}
