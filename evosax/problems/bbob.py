"""Implementation of BBOB functions provided in Hansen et al. (2009)."""

from typing import Optional
import chex
import jax
import jax.numpy as jnp
from functools import partial
from evosax.utils.visualizer_2d import BBOBVisualizer
from .bbob_helpers import (
    get_rotation,
    lambda_alpha_trafo,
    oscillation_trafo,
    asymmetry_trafo,
    boundary_penalty,
)


class BBOBFitness(object):
    """BBOB Functions Benchmark Task.
    Functions from Hansen et al. (2009)
    'Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions'
    Link: https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf
    """

    def __init__(
        self,
        fn_name: str = "Sphere",
        num_dims: int = 2,
        seed_id: int = 1,
        n_devices: int = 1,
    ):
        self.num_dims = num_dims
        # Create rotation matrices for non-separability
        rng = jax.random.PRNGKey(seed_id)
        self.seed_id = seed_id
        self.R, self.Q = self.get_rotation_matrices(rng)
        self.fn_name = fn_name
        self.fn = BBOB_fns[self.fn_name]
        self.n_devices = n_devices

    def rollout(
        self,
        rng: chex.PRNGKey,
        eval_params: chex.Array,
        R: Optional[chex.Array] = None,
        Q: Optional[chex.Array] = None,
        noise_std: float = 0.0,
    ) -> chex.Array:
        """Batch evaluate the proposal points."""
        if self.n_devices > 1:
            return self.rollout_pmap(rng, eval_params, R, Q, noise_std)
        else:
            return self.rollout_single(rng, eval_params, R, Q, noise_std)

    def rollout_pmap(
        self,
        rng: chex.PRNGKey,
        eval_params: chex.Array,
        R: chex.Array,
        Q: chex.Array,
        noise_std: float,
    ):
        """Evaluate the proposal points in parallel across devices."""
        # split the rng into n_devices
        rngs = jax.random.split(rng, self.n_devices)
        # pmap over the devices
        val = jax.pmap(self.rollout_single, in_axes=(0, 0, None, None, None))(
            rngs, eval_params, R, Q, noise_std
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def rollout_single(
        self,
        rng: chex.PRNGKey,
        eval_params: chex.Array,
        R: Optional[chex.Array] = None,
        Q: Optional[chex.Array] = None,
        noise_std: float = 0.0,
    ) -> chex.Array:
        """Batch evaluate the proposal points."""
        if R is None:
            R, Q = self.R, self.Q
        # vmap over population batch dimension
        val = jax.vmap(self.fn, in_axes=(0, None, None))(eval_params, R, Q)
        eval_noise = jax.random.normal(rng, shape=val.shape) * noise_std
        return val + eval_noise

    def get_rotation_matrices(self, rng: chex.PRNGKey):
        """Sample two rotation matrices."""
        rng_q, rng_r = jax.random.split(rng)
        R = get_rotation(rng_r, self.num_dims)
        Q = get_rotation(rng_q, self.num_dims)
        return R, Q

    def visualize(self, plot_log_fn: bool = False):
        """Visualize fitness landscape in 2D case."""
        assert self.num_dims == 2
        visualizer = BBOBVisualizer(
            None, None, self.fn_name, self.fn_name, True, plot_log_fn
        )
        visualizer.plot_contour_3d(save=False)


def Sphere(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Sphere fn - separable (p.5; Hansen et al., 2009)."""
    out = arr * arr
    return jnp.sum(out)


def EllipsoidalOriginal(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Ellipsoidal Original fn - separable (p.10; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z_vec = jax.vmap(oscillation_trafo)(arr)

    def get_val(i):
        exp = jax.lax.select(dim > 1, 6.0 * i / (dim - 1), 6.0)
        s = 10**exp * z_vec[i] * z_vec[i]
        return s

    out = jax.vmap(get_val)(jnp.arange(dim))
    return jnp.sum(out)


def RastriginOriginal(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Rastrigin fn - separable (p.15; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z = asymmetry_trafo(jax.vmap(oscillation_trafo)(arr), 0.2, dim)
    z = jnp.matmul(lambda_alpha_trafo(10.0, dim), z)
    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = z * z
    return 10 * (dim - jnp.sum(out_1)) + jnp.sum(out_2)


def BuecheRastrigin(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Bueche-Rastrigin fn - separable (p.20; Hansen et al., 2009)."""
    dim = arr.shape[0]
    t = jax.vmap(oscillation_trafo)(arr)

    def calculate_si(max_dims: int, vector: chex.Array, num_dims: int) -> chex.Array:
        def get_val(i):
            s = jax.lax.select(
                num_dims > 1, 10 ** (0.5 * (i / (num_dims - 1.0))), 10**0.5
            )
            extra_cond = jnp.logical_and(i % 2 == 0, vector[i] > 0)
            s = jax.lax.select(extra_cond, s * 10, s)
            return s

        return jax.vmap(get_val, in_axes=0)(jnp.arange(max_dims))

    L = calculate_si(dim, arr, dim) * t
    out1 = jnp.cos(2 * jnp.pi * L)
    out2 = L * L
    t1 = 10 * (dim - jnp.sum(out1 * dim))
    t2 = jnp.sum(out2 * dim)
    t3 = 100 * boundary_penalty(arr, dim)
    return t1 + t2 + t3


def LinearSlope(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Linear Slope fn - separable (p.25; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z = jnp.matmul(R, arr)

    def get_val(i):
        s = 10 ** jax.lax.select(dim > 1, i / (dim - 1.0), 1.0)
        z_opt = 5 * jnp.sum(jnp.abs(R[i, :]))
        return s * (z_opt - z[i])

    out = jax.vmap(get_val)(jnp.arange(dim))
    return jnp.sum(out)


def AttractiveSector(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Attractive Sector fn - low/mod condition (p.30; Hansen et al., 2009)."""
    dim = arr.shape[0]
    x_opt = jnp.array([1 * (i % 2 == 0) - 1 * (i % 2 != 0) for i in range(dim)])
    z_vec = jnp.matmul(R, arr - x_opt)
    z_vec = jnp.matmul(lambda_alpha_trafo(10.0, dim), z_vec)
    z_vec = jnp.matmul(Q, z_vec)

    def get_val(i):
        z = z_vec[i]
        s = jax.lax.select(z * x_opt[i] > 0, 100, 1)
        return (s * z) ** 2

    out = jax.vmap(get_val)(jnp.arange(dim))
    res = jnp.sum(out)
    return jnp.power(oscillation_trafo(res), 0.9)


def StepEllipsoidal(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Step Ellipsoidal fn - low/mod condition (p.35; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z_hat = jnp.matmul(R, arr)
    z_hat = jnp.matmul(lambda_alpha_trafo(10.0, dim), z_hat)
    z_tilde = jnp.array(
        [
            jnp.floor(0.5 + z) * (z > 0.5) + (jnp.floor(0.5 + 10 * z)) * (z <= 0.5)
            for z in z_hat
        ]
    )
    z_tilde = jnp.matmul(Q, z_tilde)

    def get_val(i):
        exponent = jax.lax.select(dim > 1.0, 2.0 * i / (dim - 1.0), 2.0)
        return 10.0**exponent * z_tilde[i] ** 2

    out = jax.vmap(get_val)(jnp.arange(dim))
    res = jnp.sum(out)
    value = jnp.maximum(jnp.abs(z_hat[0]) / 1000, res)
    return 0.1 * value + boundary_penalty(arr, dim)


def RosenbrockOriginal(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """RosenbrockOriginal fn - low/mod condition (p.40; Hansen et al., 2009)."""
    dim = arr.shape[0]
    # Optimum is at origin!
    z = jnp.maximum(1.0, (dim**0.5) / 8.0) * arr + 1
    out = jnp.array(
        [100.0 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2 for i in range(dim - 1)]
    )
    return jnp.sum(out)


def RosenbrockRotated(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """RosenbrockRotated fn - low/mod condition (p.45; Hansen et al., 2009)."""
    dim = arr.shape[0]
    r_x = jnp.matmul(R, arr)
    z = jnp.maximum(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * jnp.ones((dim,))
    out = jnp.array(
        [100.0 * (z[i] ** 2 - z[i + 1]) ** 2 + (z[i] - 1) ** 2 for i in range(dim - 1)]
    )
    return jnp.sum(out)


def EllipsoidalRotated(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Ellipsoidal Rotated fn - high condition (p.50; Hansen et al., 2009)."""
    dim = arr.shape[0]
    r_x = jnp.matmul(R, arr)
    z_vec = jax.vmap(oscillation_trafo)(r_x)

    def get_val(i):
        exp = jax.lax.select(dim > 1, 6.0 * i / (dim - 1), 6.0)
        s = 10**exp * z_vec[i] * z_vec[i]
        return s

    out = jax.vmap(get_val)(jnp.arange(dim))
    return jnp.sum(out)


def Discus(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Discus fn - high condition (p.55; Hansen et al., 2009)."""
    r_x = jnp.matmul(R, arr)
    z_vec = jax.vmap(oscillation_trafo)(r_x)
    out = jnp.array([z * z for z in z_vec[1:]])
    return 10**6 * z_vec[0] * z_vec[0] + jnp.sum(out)


def BentCigar(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Bent Cigar fn - high condition (p.60; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z_vec = jnp.matmul(R, arr)
    z_vec = asymmetry_trafo(z_vec, 0.5, dim)
    z_vec = jnp.matmul(R, z_vec)
    out = z_vec[1:] ** 2
    return z_vec[0] ** 2 + 10**6 * jnp.sum(out)


def SharpRidge(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Sharp Ridge fn - high condition (p.65; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z_vec = jnp.matmul(R, arr)
    z_vec = jnp.matmul(lambda_alpha_trafo(10, dim), z_vec)
    z_vec = jnp.matmul(Q, z_vec)
    out = z_vec[1:] ** 2
    return z_vec[0] ** 2 + 100 * jnp.sum(out) ** 0.5


def DifferentPowers(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Different Powers fn - high condition (p.70; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z = jnp.matmul(R, arr)

    def get_val(i):
        exp = (2 + 4 * i / (dim - 1)) * (dim > 1) + (dim <= 1) * 6
        s = jnp.abs(z[i]) ** exp
        return s

    out = jax.vmap(get_val)(jnp.arange(dim))
    return jnp.sum(out) ** 0.5


def RastriginRotated(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Rastrigin fn - multi-modal (p.75; Hansen et al., 2009)."""
    dim = arr.shape[0]
    z = jnp.matmul(R, arr)
    z = asymmetry_trafo(jax.vmap(oscillation_trafo)(z), 0.2, dim)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha_trafo(10.0, dim), z)
    z = jnp.matmul(R, z)
    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = z * z
    return 10 * (dim - jnp.sum(out_1)) + jnp.sum(out_2)


def Weierstrass(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Weierstrass fn - multi-modal (p.80; Hansen et al., 2009)."""
    k_order = 12
    dim = arr.shape[0]
    z = jnp.matmul(R, arr)
    z = jax.vmap(oscillation_trafo)(z)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha_trafo(1.0 / 100.0, dim), z)
    f0 = jnp.array([0.5**k * jnp.cos(jnp.pi * 3**k) for k in range(k_order)]).sum()

    def get_val(i):
        s = 0
        for k in range(k_order):
            s += 0.5**k * jnp.cos(2 * jnp.pi * (3**k) * (z[i] + 0.5))
        return s

    out = jax.vmap(get_val)(jnp.arange(dim))
    s = jnp.sum(out)
    return 10 * (s / dim - f0) ** 3 + 10 * boundary_penalty(arr, dim) / dim


def SchaffersF7(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """SchaffersF7 fn - multi-modal (p.85; Hansen et al., 2009)."""
    dim = arr.shape[0]
    if dim == 1:
        return 0.0
    z = jnp.matmul(R, arr)
    z = asymmetry_trafo(z, 0.5, dim)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha_trafo(10.0, dim), z)

    def get_val_arr(i):
        s_arr = (z[i] ** 2 + z[i + 1] ** 2) ** 0.5
        return s_arr

    s_arr = jax.vmap(get_val_arr, in_axes=0)(jnp.arange(dim - 1))

    def get_val(i):
        return s_arr[i] ** 0.5 + (s_arr[i] ** 0.5) * jnp.sin(50 * s_arr[i] ** 0.2) ** 2

    out = jax.vmap(get_val)(jnp.arange(dim - 1))
    s = jnp.sum(out)
    return (s / (dim - 1.0)) ** 2 + 10 * boundary_penalty(arr, dim)


def SchaffersF7IllConditioned(
    arr: chex.Array, R: chex.Array, Q: chex.Array
) -> chex.Array:
    """SchaffersF7 ill condition - multi-modal (p.90; Hansen et al., 2009)."""
    dim = arr.shape[0]
    if dim == 1:
        return 0.0
    z = jnp.matmul(R, arr)
    z = asymmetry_trafo(z, 0.5, dim)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha_trafo(1000.0, dim), z)

    def get_val_arr(i):
        s_arr = (z[i] ** 2 + z[i + 1] ** 2) ** 0.5
        return s_arr

    s_arr = jax.vmap(get_val_arr, in_axes=0)(jnp.arange(dim - 1))

    def get_val(i):
        return s_arr[i] ** 0.5 + (s_arr[i] ** 0.5) * jnp.sin(50 * s_arr[i] ** 0.2) ** 2

    out = jax.vmap(get_val)(jnp.arange(dim - 1))
    s = jnp.sum(out)
    return (s / (dim - 1.0)) ** 2 + 10 * boundary_penalty(arr, dim)


def GriewankRosenbrock(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Griewank-Rosenbrock fn - multi-modal weak (p.95; Hansen et al., 2009)."""
    dim = arr.shape[0]
    r_x = jnp.matmul(R, arr)
    # Original BBOB: max(1.0, (dim ** 0.5)/8) * r_x + 0.5 * jnp.ones((dim,))
    z_arr = jnp.maximum(1.0, (dim**0.5) / 8.0) * r_x + jnp.ones((dim,))
    s_arr = jnp.zeros(dim)

    def get_val_arr(i):
        s_arr = 100.0 * (z_arr[i] ** 2 - z_arr[i + 1]) ** 2 + (z_arr[i] - 1) ** 2
        return s_arr

    s_arr = jax.vmap(get_val_arr)(jnp.arange(dim - 1))

    def get_val(i):
        val = s_arr[i] / 4000.0 - jnp.cos(s_arr[i])
        return val

    out = jax.vmap(get_val)(jnp.arange(dim - 1))
    total = jnp.sum(out)
    return (10.0 * total) / (dim - 1) + 10


def Schwefel(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Schwefel fn - multi-modal weak (p.100; Hansen et al., 2009)."""
    dim = arr.shape[0]
    bernoulli_arr = jnp.array([jnp.power(-1, i + 1) for i in range(dim)])
    x_opt = 4.2096874633 / 2.0 * bernoulli_arr
    x_hat = 2.0 * (bernoulli_arr * arr)  # element-wise multiply
    z_hat = jnp.zeros(dim)
    z_hat = z_hat.at[0].set(x_hat[0])

    def get_val(i):
        val = x_hat[i] + 0.25 * (x_hat[i - 1] - 2 * jnp.abs(x_opt[i - 1]))
        return val

    vals = jax.vmap(get_val)(jnp.arange(1, dim))
    z_hat = z_hat.at[1:].set(vals)
    z_vec = 100 * (
        jnp.matmul(lambda_alpha_trafo(10, dim), z_hat - 2 * jnp.abs(x_opt))
        + 2 * jnp.abs(x_opt)
    )
    out = z_vec * jnp.sin(jnp.abs(z_vec) ** 0.5)
    total = jnp.sum(out)
    return (
        -(total / (100.0 * dim))
        + 4.189828872724339
        + 100 * boundary_penalty(z_vec / 100, dim)
    )


def Gallagher101Me(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Gallagher 101 peaks - multi-modal weak (p.105; Hansen et al., 2009)."""
    dim = arr.shape[0]
    num_optima = 101

    def get_optima(i):
        vec = jnp.zeros(dim)
        for j in range(dim):
            alpha = ((i - 1) * dim + j + 1.0) / (dim * num_optima + 2.0)
            vec = vec.at[j].set(-5 + 10 * alpha)
        return vec * (i != 0) + jnp.zeros([dim]) * (i == 0)

    optima_list = jax.vmap(get_optima)(jnp.arange(num_optima))

    def get_c_val(i):
        alpha = 1000.0 ** (2.0 * (i - 1) / (num_optima - 2))
        c_mat = lambda_alpha_trafo(alpha, dim) / (alpha**0.25)
        return c_mat * (i != 0) + lambda_alpha_trafo(1000, dim) * (i == 0)

    c_list = jax.vmap(get_c_val)(jnp.arange(num_optima))
    max_value = jnp.array([-1.0])

    for i in range(num_optima):
        w = 10 * (i == 0) + (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0)) * (i != 0)
        diff = jnp.matmul(R, (arr - optima_list[i]).reshape(-1, 1))
        e = jnp.matmul(diff.transpose(), jnp.matmul(c_list[i], diff))
        max_value = jnp.maximum(max_value, w * jnp.exp(-e / (2.0 * dim)))
    return oscillation_trafo(10.0 - max_value.squeeze()) ** 2 + boundary_penalty(
        arr, dim
    )


def Gallagher21Hi(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Gallagher 21 peaks fn - multi-modal weak (p.110; Hansen et al., 2009)."""
    dim = arr.shape[0]
    num_optima = 21

    def get_optima(i):
        vec = jnp.zeros(dim)
        for j in range(dim):
            alpha = ((i - 1) * dim + j + 1.0) / (dim * num_optima + 2.0)
            vec = vec.at[j].set(-5 + 10 * alpha)
        return vec * (i != 0) + jnp.zeros([dim]) * (i == 0)

    optima_list = jax.vmap(get_optima)(jnp.arange(num_optima))

    def get_c_val(i):
        alpha = 1000.0 ** (2.0 * (i - 1) / (num_optima - 2))
        c_mat = lambda_alpha_trafo(alpha, dim) / (alpha**0.25)
        return c_mat * (i != 0) + lambda_alpha_trafo(1000, dim) * (i == 0)

    c_list = jax.vmap(get_c_val)(jnp.arange(num_optima))
    max_value = jnp.array([-1.0])

    for i in range(num_optima):
        w = 10 * (i == 0) + (1.1 + 8.0 * (i - 1.0) / (num_optima - 2.0)) * (i != 0)
        diff = jnp.matmul(R, (arr - optima_list[i]).reshape(-1, 1))
        e = jnp.matmul(diff.transpose(), jnp.matmul(c_list[i], diff))
        max_value = jnp.maximum(max_value, w * jnp.exp(-e / (2.0 * dim)))
    return oscillation_trafo(10.0 - max_value.squeeze()) ** 2 + boundary_penalty(
        arr, dim
    )


def Katsuura(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Katsuura fn - multi-modal weak (p.115; Hansen et al., 2009)."""
    # NOTE: Numerically instable - need higher precision float64
    # from jax.config import config
    # config.update("jax_enable_x64", True)
    dim = arr.shape[0]
    r_x = jnp.matmul(R, arr)
    z_vec = jnp.matmul(lambda_alpha_trafo(100.0, dim), r_x)
    z_vec = jnp.matmul(Q, z_vec)

    def get_val(i):
        s = 0.0
        for j in range(1, 33):
            s += jnp.abs(2**j * z_vec[i] - jnp.round(2**j * z_vec[i])) / 2**j
        return (1 + (i + 1) * s) ** (10.0 / dim**1.2)

    out = jax.vmap(get_val)(jnp.arange(dim))
    mask = jnp.arange(dim) < dim
    coeff = 1.0 - mask  # 1st to zero and add 1 for multiply
    prod = jnp.prod(out * mask + coeff)
    return (10.0 / dim**2) * prod - 10.0 / dim**2 + boundary_penalty(arr, dim)


def Lunacek(arr: chex.Array, R: chex.Array, Q: chex.Array) -> chex.Array:
    """Lunacek fn - multi-modal weak (p.120; Hansen et al., 2009)."""
    dim = arr.shape[0]
    mu0 = 2.5
    s = 1.0 - 1.0 / (2.0 * (dim + 20.0) ** 0.5 - 8.2)
    mu1 = -(((mu0**2 - 1) / s) ** 0.5)
    x_opt = jnp.array([mu0 / 2] * dim)
    x_hat = 2 * arr * jnp.sign(x_opt)
    x_vec = x_hat - mu0
    x_vec = jnp.matmul(R, x_vec)
    z_vec = jnp.matmul(lambda_alpha_trafo(100, dim), x_vec)
    z_vec = jnp.matmul(Q, z_vec)

    s1 = jnp.sum((x_hat - mu0) ** 2)
    s2 = jnp.sum((x_hat - mu1) ** 2)
    s3 = jnp.sum(jnp.cos(2 * jnp.pi * z_vec))
    return (
        jnp.minimum(s1, dim + s * s2)
        + 10.0 * (dim - s3)
        + 10**4 * boundary_penalty(arr, dim)
    )


BBOB_fns = {
    # Part 1: Separable functions
    "Sphere": Sphere,
    "EllipsoidalOriginal": EllipsoidalOriginal,
    "RastriginOriginal": RastriginOriginal,
    "BuecheRastrigin": BuecheRastrigin,
    "LinearSlope": LinearSlope,
    # Part 2: Functions with low or moderate conditions
    "AttractiveSector": AttractiveSector,
    "StepEllipsoidal": StepEllipsoidal,
    "RosenbrockOriginal": RosenbrockOriginal,
    "RosenbrockRotated": RosenbrockRotated,
    # Part 3: Functions with high conditioning and unimodal
    "EllipsoidalRotated": EllipsoidalRotated,
    "Discus": Discus,
    "BentCigar": BentCigar,
    "SharpRidge": SharpRidge,
    "DifferentPowers": DifferentPowers,
    # Part 4: Multi-modal functions with adequate global structure
    "RastriginRotated": RastriginRotated,
    "Weierstrass": Weierstrass,
    "SchaffersF7": SchaffersF7,
    "SchaffersF7IllConditioned": SchaffersF7IllConditioned,
    "GriewankRosenbrock": GriewankRosenbrock,
    # Part 5: Multi-modal functions with weak global structure
    "Schwefel": Schwefel,
    "Katsuura": Katsuura,
    "Lunacek": Lunacek,
    "Gallagher101Me": Gallagher101Me,
    "Gallagher21Hi": Gallagher21Hi,
}
