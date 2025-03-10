"""Blackbox Optimization Benchmarking Functions.

This module implements the core 24 functions used in the BBOB suite from [1].

[1] https://inria.hal.science/inria-00362633
[2] https://coco-platform.org/
[3] https://numbbo.github.io/temp-doc-bbob/bbob/def.html
"""

import jax
import jax.numpy as jnp


def lambda_alpha(alpha: float, max_num_dims: int, num_dims: int) -> jax.Array:
    """Masked lambda alpha matrix ([1], p. 5)."""
    mask = jnp.arange(max_num_dims) < num_dims

    exp = (
        jnp.where(num_dims > 1, 0.5 * jnp.arange(max_num_dims) / (num_dims - 1), 0.5)
        * mask
    )
    return jnp.diag(jnp.power(alpha, exp))


def transform_osz(element: float) -> jax.Array:
    """Oscillation transformation function ([1], p. 6)."""
    x_hat = jnp.where(element == 0.0, 0.0, jnp.log(jnp.abs(element)))
    c_1 = jnp.where(element > 0.0, 10.0, 5.5)
    c_2 = jnp.where(element > 0.0, 7.9, 3.1)
    return jnp.sign(element) * jnp.exp(
        x_hat + 0.049 * (jnp.sin(c_1 * x_hat) + jnp.sin(c_2 * x_hat))
    )


def transform_asy(x: jax.Array, beta: float, num_dims: int) -> jax.Array:
    """Asymmetry transformation function ([1], p. 6)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    exp = (
        1
        + beta
        * jnp.where(num_dims > 1, jnp.arange(max_num_dims) / (num_dims - 1), 1.0)
        * jnp.sqrt(x)
        * mask
    )
    return jnp.where(x > 0.0, jnp.power(x, exp), x)


def f_pen(x: jax.Array, num_dims: int) -> jax.Array:
    """Boundary penalty ([1], p. 5)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    out = jnp.abs(x) - 5.0
    return jnp.sum(jnp.square(jnp.maximum(0.0, out * mask)))


def sphere(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Sphere Function ([1], p. 6)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = x - x_opt

    out = jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def ellipsoidal(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Ellipsoidal Function ([1], p. 6)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = transform_osz(x - x_opt)

    exp = (
        jnp.where(num_dims > 1, 6.0 * jnp.arange(max_num_dims) / (num_dims - 1), 6.0)
        * mask
    )
    out = jnp.power(10, exp) * jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def rastrigin(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Rastrigin Function ([1], p. 6)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = transform_asy(transform_osz(x - x_opt), 0.2, num_dims)
    z = jnp.matmul(lambda_alpha(10.0, max_num_dims, num_dims), z)

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)
    return (
        10 * (num_dims - jnp.sum(out_1 * mask)) + jnp.sum(out_2 * mask),
        jnp.array(0.0),
    )


def bueche_rastrigin(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Bueche-Rastrigin Function ([1], p. 7)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = transform_osz(x - x_opt)

    exp = (
        jnp.where(num_dims > 1, 0.5 * jnp.arange(max_num_dims) / (num_dims - 1), 0.5)
        * mask
    )
    cond = (z > 0.0) & (jnp.arange(max_num_dims) % 2 == 1)
    s = jnp.where(cond, jnp.power(10, exp + 1), jnp.power(10, exp))

    z = s * z

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)
    return (
        10 * (num_dims - jnp.sum(out_1 * mask)) + jnp.sum(out_2 * mask),
        100 * f_pen(x, num_dims),
    )


def linear_slope(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Linear Slope ([1], p. 7)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    # x_opt = 5.0 * (2 * jax.random.bernoulli(subkey, shape=(max_num_dims,)) - 1)
    x_opt = jnp.where(x_opt > 0.0, 5.0, -5.0)

    z = jnp.where(x * x_opt < 25.0, x, x_opt)

    exp = jnp.where(num_dims > 1, jnp.arange(max_num_dims) / (num_dims - 1), 0.5) * mask
    s = jnp.sign(x_opt) * jnp.power(10, exp)

    out = 5 * jnp.abs(s) - s * z
    return jnp.sum(out * mask), jnp.array(0.0)


def attractive_sector(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Attractive Sector Function ([1], p. 8)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = jnp.matmul(lambda_alpha(10.0, max_num_dims, num_dims), z)
    z = jnp.matmul(Q, z)

    s = jnp.where(z * x_opt > 0.0, 100.0, 1.0)

    out = jnp.sum(jnp.square(s * z) * mask)
    return jnp.power(transform_osz(out), 0.9), jnp.array(0.0)


def step_ellipsoidal(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Step Ellipsoidal Function ([1], p. 8)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z_hat = jnp.matmul(R, x - x_opt)
    z_hat = jnp.matmul(lambda_alpha(10.0, max_num_dims, num_dims), z_hat)

    z_tilde = jnp.where(
        jnp.abs(z_hat) > 0.5,
        jnp.floor(0.5 + z_hat),
        jnp.floor(0.5 + 10 * z_hat) / 10,
    )

    z = jnp.matmul(Q, z_tilde)

    exp = (
        jnp.where(num_dims > 1, 2.0 * jnp.arange(max_num_dims) / (num_dims - 1.0), 2.0)
        * mask
    )
    out = jnp.sum(100.0 * jnp.power(10.0, exp) * jnp.square(z) * mask)
    return 0.1 * jnp.maximum(jnp.abs(z_hat[0]) / 1e4, out), f_pen(x, num_dims)


def rosenbrock(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Rosenbrock Function, original ([1], p. 8).

    Exceptionally, here x_opt must be in [-3, 3].
    """
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    z = jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * (x - x_opt) + 1.0
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * (z_i**2 - z_ip1) ** 2 + (z_i - 1) ** 2
    return jnp.sum(out * mask), jnp.array(0.0)


def rosenbrock_rotated(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Rosenbrock Function, rotated ([1], p. 9)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    z = jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * jnp.matmul(R, x) + 0.5
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    out = 100.0 * (z_i**2 - z_ip1) ** 2 + (z_i - 1) ** 2
    return jnp.sum(out * mask), jnp.array(0.0)


def ellipsoidal_rotated(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Ellipsoidal Function ([1], p. 9)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = transform_osz(z)

    exp = (
        jnp.where(num_dims > 1, 6.0 * jnp.arange(max_num_dims) / (num_dims - 1), 6.0)
        * mask
    )
    out = jnp.power(10, exp) * jnp.square(z)
    return jnp.sum(out * mask), jnp.array(0.0)


def discus(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Discus Function ([1], p. 9)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = transform_osz(z)

    z_squared = jnp.square(z)
    out = z_squared.at[0].multiply(10**6)
    return jnp.sum(out * mask), jnp.array(0.0)


def bent_cigar(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Bent Cigar Function ([1], p. 10)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = transform_asy(z, 0.5, num_dims)
    z = jnp.matmul(R, z)

    z_squared = jnp.square(z)
    out = z_squared.at[1:].multiply(10**6)
    return jnp.sum(out * mask), jnp.array(0.0)


def sharp_ridge(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Sharp Ridge Function ([1], p. 10)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    z = jnp.matmul(R, x - x_opt)
    z = jnp.matmul(lambda_alpha(10, max_num_dims, num_dims), z)
    z = jnp.matmul(Q, z)

    z_squared = jnp.square(z)
    return z_squared[0] + 100 * jnp.sqrt(jnp.sum(z_squared[1:] * mask)), jnp.array(0.0)


def different_powers(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Different Powers Function ([1], p. 10)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)

    exp = (
        jnp.where(
            num_dims > 1, 2.0 + 4.0 * jnp.arange(max_num_dims) / (num_dims - 1), 6.0
        )
        * mask
    )
    out = jnp.power(jnp.abs(z), exp)
    return jnp.sqrt(jnp.sum(out * mask)), jnp.array(0.0)


def rastrigin_rotated(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Rastrigin Function ([1], p. 11)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = transform_asy(transform_osz(z), 0.2, num_dims)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha(10.0, max_num_dims, num_dims), z)
    z = jnp.matmul(R, z)

    out_1 = jnp.cos(2 * jnp.pi * z)
    out_2 = jnp.square(z)
    return (
        10 * (num_dims - jnp.sum(out_1 * mask)) + jnp.sum(out_2 * mask),
        jnp.array(0.0),
    )


def weierstrass(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Weierstrass Function ([1], p. 11)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    z = jnp.matmul(R, x - x_opt)
    z = transform_osz(z)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha(0.01, max_num_dims, num_dims), z)
    z = jnp.matmul(R, z)

    k_order = 12
    half_pow_k = jnp.power(0.5, jnp.arange(k_order))
    three_pow_k = jnp.power(3, jnp.arange(k_order))
    f_0 = jnp.sum(half_pow_k * jnp.cos(jnp.pi * three_pow_k))

    out = jnp.sum(
        half_pow_k
        * jnp.cos(2 * jnp.pi * three_pow_k * (z[:, None] + 0.5))
        * mask[:, None]
    )
    return 10 * (out / num_dims - f_0) ** 3, 10 * f_pen(x, num_dims) / num_dims


def schaffers_f7(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Schaffers F7 Function ([1], p. 11)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    if max_num_dims == 1:
        return 0.0

    z = jnp.matmul(R, x - x_opt)
    z = transform_asy(z, 0.5, num_dims)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha(10.0, max_num_dims, num_dims), z)

    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]
    s = jnp.sqrt(z_i**2 + z_ip1**2)

    out = jnp.sum((jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2) * mask)
    return (out / (num_dims - 1.0)) ** 2, 10 * f_pen(x, num_dims)


def schaffers_f7_ill_cond(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Schaffers F7 Function, moderately ill-conditioned ([1], p. 12)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    if max_num_dims == 1:
        return 0.0

    z = jnp.matmul(R, x - x_opt)
    z = transform_asy(z, 0.5, num_dims)
    z = jnp.matmul(Q, z)
    z = jnp.matmul(lambda_alpha(1000.0, max_num_dims, num_dims), z)

    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]
    s = jnp.sqrt(z_i**2 + z_ip1**2)

    out = jnp.sum((jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50 * s**0.2) ** 2) * mask)
    return (out / (num_dims - 1.0)) ** 2, 10 * f_pen(x, num_dims)


def griewank_rosenbrock(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Composite Griewank-Rosenbrock Function F8F2 ([1], p. 12)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims - 1) < (num_dims - 1)

    z = jnp.maximum(1.0, jnp.sqrt(num_dims) / 8.0) * jnp.matmul(R, x) + 0.5
    z_i = z[:-1]
    z_ip1 = jnp.roll(z, -1)[:-1]

    s = 100.0 * (z_i**2 - z_ip1) ** 2 + (z_i - 1) ** 2
    out = s / 4000.0 - jnp.cos(s)
    return 10.0 * jnp.sum(out * mask) / (num_dims - 1) + 10, jnp.array(0.0)


def schwefel(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Schwefel Function ([1], p. 12)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    bernoulli = jnp.where(x_opt > 0.0, 1.0, -1.0)
    x_opt = 4.2096874633 * bernoulli / 2

    x_hat = 2.0 * bernoulli * x
    x_hat_i = x_hat
    x_hat_im1 = jnp.roll(x_hat, 1).at[0].set(0.0)
    x_opt_im1 = jnp.roll(x_opt, 1).at[0].set(0.0)
    z_hat = x_hat_i + 0.25 * (x_hat_im1 - 2 * jnp.abs(x_opt_im1))
    z = 100 * (
        jnp.matmul(lambda_alpha(10, max_num_dims, num_dims), z_hat - 2 * jnp.abs(x_opt))
        + 2 * jnp.abs(x_opt)
    )

    out = z * jnp.sin(jnp.sqrt(jnp.abs(z)))
    return (
        -(jnp.sum(out * mask) / (100.0 * num_dims)) + 4.189828872724339,
        100 * f_pen(z / 100, num_dims),
    )


def gallagher_101_me(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Gallagher's Gaussian 101-me Peaks Function ([1], p. 13)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    num_optima = 101
    key = jax.random.key(0)
    key = jax.random.fold_in(key, Q[0, 0])

    w = jnp.where(
        jnp.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (jnp.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha = jnp.zeros(num_optima)
    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha_permuted = jax.random.permutation(subkey, alpha_set)
    alpha = alpha.at[0].set(1000.0)
    alpha = alpha.at[1:].set(alpha_permuted)
    c = jax.vmap(
        lambda alpha: lambda_alpha(alpha, max_num_dims, num_dims) / alpha**0.25
    )(alpha)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)

    def permute_diag(c_i, key):
        perm = jax.random.choice(
            key, jnp.arange(max_num_dims), shape=(max_num_dims,), replace=False, p=mask
        )
        diag = jnp.diag(c_i)
        new_diag = diag[perm]
        return c_i.at[jnp.arange(max_num_dims), jnp.arange(max_num_dims)].set(new_diag)

    c = jax.vmap(permute_diag)(c, keys)

    key, subkey = jax.random.split(key)
    y = jax.random.uniform(
        subkey,
        shape=(
            num_optima,
            max_num_dims,
        ),
        minval=-5.0,
        maxval=5.0,
    )
    y = y.at[0].set(x_opt) * mask

    def f(c_i, y_i, w_i):
        out = jnp.matmul(R, x - y_i)
        out = jnp.matmul(c_i, out)
        out = jnp.matmul(jnp.transpose(R), out)
        out = jnp.dot(x - y_i, out * mask)
        return w_i * jnp.exp(-out / (2 * num_dims))

    out = jax.vmap(f)(c, y, w)
    return jnp.square(transform_osz(10.0 - jnp.max(out))), f_pen(x, num_dims)


def gallagher_21_hi(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Gallagher's Gaussian 21-hi Peaks Function ([1], p. 13)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    x_opt *= 3.92 / 4.0

    num_optima = 21
    key = jax.random.key(0)
    key = jax.random.fold_in(key, Q[0, 0])

    w = jnp.where(
        jnp.arange(num_optima) == 0,
        10.0,
        1.1 + 8.0 * (jnp.arange(num_optima) - 1.0) / (num_optima - 2.0),
    )

    alpha = jnp.zeros(num_optima)
    alpha_set = jnp.power(1000.0, 2.0 * jnp.arange(num_optima - 1) / (num_optima - 2))
    key, subkey = jax.random.split(key)
    alpha_permuted = jax.random.permutation(subkey, alpha_set)
    alpha = alpha.at[0].set(1000.0**2)
    alpha = alpha.at[1:].set(alpha_permuted)
    c = jax.vmap(
        lambda alpha: lambda_alpha(alpha, max_num_dims, num_dims) / alpha**0.25
    )(alpha)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_optima)

    def permute_diag(c_i, key):
        perm = jax.random.choice(
            key, jnp.arange(max_num_dims), shape=(max_num_dims,), replace=False, p=mask
        )
        diag = jnp.diag(c_i)
        new_diag = diag[perm]
        return c_i.at[jnp.arange(max_num_dims), jnp.arange(max_num_dims)].set(new_diag)

    c = jax.vmap(permute_diag)(c, keys)

    key, subkey = jax.random.split(key)
    y = jax.random.uniform(
        subkey,
        shape=(
            num_optima,
            max_num_dims,
        ),
        minval=-4.9,
        maxval=4.9,
    )
    y = y.at[0].set(x_opt) * mask

    def f(c_i, y_i, w_i):
        out = jnp.matmul(R, x - y_i)
        out = jnp.matmul(c_i, out)
        out = jnp.matmul(jnp.transpose(R), out)
        out = jnp.dot(x - y_i, out * mask)
        return w_i * jnp.exp(-out / (2 * num_dims))

    out = jax.vmap(f)(c, y, w)
    return jnp.square(transform_osz(10.0 - jnp.max(out))), f_pen(x, num_dims)


def katsuura(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Katsuura Function ([1], p. 13)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    # num_terms = 32 in Hansen et al. (2010), but set to 30 to avoid overflow
    num_terms = 30

    z = jnp.matmul(R, x - x_opt)
    z = jnp.matmul(lambda_alpha(100.0, max_num_dims, num_dims), z)
    z = jnp.matmul(Q, z)

    two_pow_j = jnp.power(2, jnp.arange(1, num_terms + 1))
    sum = jnp.sum(
        jnp.abs(two_pow_j * z[:, None] - jnp.round(two_pow_j * z[:, None])) / two_pow_j,
        axis=1,
    )
    prod = jnp.prod(1 + jnp.arange(1, max_num_dims + 1) * sum * mask)
    return (
        (10.0 / num_dims**2) * (jnp.power(prod, 10.0 / num_dims**1.2) - 1.0),
        f_pen(x, num_dims),
    )


def lunacek(
    x: jax.Array, x_opt: jax.Array, R: jax.Array, Q: jax.Array, num_dims: int
) -> jax.Array:
    """Lunacek bi-Rastrigin Function ([1], p. 14)."""
    max_num_dims = x.shape[0]
    mask = jnp.arange(max_num_dims) < num_dims

    mu_0 = 2.5
    d = 1
    s = 1.0 - 1 / (2.0 * jnp.sqrt(num_dims + 20.0) - 8.2)
    mu_1 = -jnp.sqrt((mu_0**2 - d) / s)

    x_opt = jnp.where(x_opt > 0.0, mu_0 / 2, -mu_0 / 2)
    x_hat = 2 * jnp.sign(x_opt) * x

    z = jnp.matmul(R, x_hat - mu_0)
    z = jnp.matmul(lambda_alpha(100, max_num_dims, num_dims), z)
    z = jnp.matmul(Q, z)

    s_1 = jnp.sum(jnp.square(x_hat - mu_0) * mask)
    s_2 = jnp.sum(jnp.square(x_hat - mu_1) * mask)
    s_3 = jnp.sum(jnp.cos(2 * jnp.pi * z) * mask)
    return (
        jnp.minimum(s_1, d * num_dims + s * s_2) + 10.0 * (num_dims - s_3),
        10**4 * f_pen(x, num_dims),
    )


bbob_fns = {
    # Part 1: Separable functions
    "sphere": sphere,
    "ellipsoidal": ellipsoidal,
    "rastrigin": rastrigin,
    "bueche_rastrigin": bueche_rastrigin,
    "linear_slope": linear_slope,
    # Part 2: Functions with low or moderate conditions
    "attractive_sector": attractive_sector,
    "step_ellipsoidal": step_ellipsoidal,
    "rosenbrock": rosenbrock,
    "rosenbrock_rotated": rosenbrock_rotated,
    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_rotated": ellipsoidal_rotated,
    "discus": discus,
    "bent_cigar": bent_cigar,
    "sharp_ridge": sharp_ridge,
    "different_powers": different_powers,
    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_rotated": rastrigin_rotated,
    "weierstrass": weierstrass,
    "schaffers_f7": schaffers_f7,
    "schaffers_f7_ill_cond": schaffers_f7_ill_cond,
    "griewank_rosenbrock": griewank_rosenbrock,
    # Part 5: Multi-modal functions with weak global structure
    "schwefel": schwefel,
    "gallagher_101_me": gallagher_101_me,
    "gallagher_21_hi": gallagher_21_hi,
    "katsuura": katsuura,
    "lunacek": lunacek,
}

fn_names_dict = {
    # Part 1: Separable functions
    "sphere": "Sphere Function",
    "ellipsoidal": "Ellipsoidal Function",
    "rastrigin": "Rastrigin Function",
    "bueche_rastrigin": "Büche-Rastrigin Function",
    "linear_slope": "Linear Slope",
    # Part 2: Functions with low or moderate conditioning
    "attractive_sector": "Attractive Sector Function",
    "step_ellipsoidal": "Step Ellipsoidal Function",
    "rosenbrock": "Rosenbrock Function",
    "rosenbrock_rotated": "Rosenbrock Function, rotated",
    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_rotated": "Ellipsoidal Function, rotated",
    "discus": "Discus Function",
    "bent_cigar": "Bent Cigar Function",
    "sharp_ridge": "Sharp Ridge Function",
    "different_powers": "Different Powers Function",
    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_rotated": "Rastrigin Function",
    "weierstrass": "Weierstrass Function",
    "schaffers_f7": "Schaffers F7 Function",
    "schaffers_f7_ill_cond": "Schaffers F7 Function, moderately ill-conditioned",
    "griewank_rosenbrock": "Composite Griewank-Rosenbrock Function F8F2",
    # Part 5: Multi-modal functions with weak global structure
    "schwefel": "Schwefel Function",
    "gallagher_101_me": "Gallagher's Gaussian 101-me Peaks Function",
    "gallagher_21_hi": "Gallagher's Gaussian 21-hi Peaks Function",
    "katsuura": "Katsuura Function",
    "lunacek": "Lunacek bi-Rastrigin Function",
}

fn_names_short_dict = {
    # Part 1: Separable functions
    "sphere": "Sphere",
    "ellipsoidal": "Ellipsoidal",
    "rastrigin": "Rastrigin",
    "bueche_rastrigin": "Büche-Rastrigin",
    "linear_slope": "Linear Slope",
    # Part 2: Functions with low or moderate conditioning
    "attractive_sector": "Attractive Sector",
    "step_ellipsoidal": "Step Ellipsoidal",
    "rosenbrock": "Rosenbrock",
    "rosenbrock_rotated": "Rosenbrock Rotated",
    # Part 3: Functions with high conditioning and unimodal
    "ellipsoidal_rotated": "Ellipsoidal Rotated",
    "discus": "Discus",
    "bent_cigar": "Bent Cigar",
    "sharp_ridge": "Sharp Ridge",
    "different_powers": "Different Powers",
    # Part 4: Multi-modal functions with adequate global structure
    "rastrigin_rotated": "Rastrigin Rotated",
    "weierstrass": "Weierstrass",
    "schaffers_f7": "Schaffers F7",
    "schaffers_f7_ill_cond": "Schaffers F7 Ill-cond.",
    "griewank_rosenbrock": "Griewank-Rosenbrock",
    # Part 5: Multi-modal functions with weak global structure
    "schwefel": "Schwefel",
    "gallagher_101_me": "Gallagher 101-me",
    "gallagher_21_hi": "Gallagher 21-hi",
    "katsuura": "Katsuura",
    "lunacek": "Lunacek",
}
