import chex
import jax
import jax.numpy as jnp


def get_rotation(rng: chex.PRNGKey, num_dims: int) -> chex.Array:
    """Return orthonormal rotation matrix via QR decomposition."""
    # Used to turn separable functions into non-separable
    A = jax.random.normal(rng, (num_dims, num_dims))
    Q = jnp.linalg.qr(A)[0]
    return Q


def lambda_alpha_trafo(alpha: float, num_dims: int) -> chex.Array:
    """LambdaAlpha matrix. Diagonal matrix with alpha powers."""

    def get_diag(i):
        exp = (0.5 * (i / (num_dims - 1))) * (num_dims > 1) + 0.5 * (
            num_dims <= 1
        )
        return alpha ** exp

    diag_vals = jax.vmap(get_diag, in_axes=0)(jnp.arange(num_dims))
    return jnp.diag(diag_vals)


def oscillation_trafo(element: float) -> chex.Array:
    """Oscillation trafo function for x array input & f value output.
    (p.3; Hansen et al., 2009)"""
    x_carat = jax.lax.select(element == 0, 0.0, jnp.log(jnp.abs(element)))
    c1 = jax.lax.select(element > 0, 10.0, 5.5)
    c2 = jax.lax.select(element > 0, 7.9, 3.1)
    return jnp.sign(element) * jnp.exp(
        x_carat + 0.049 * (jnp.sin(c1 * x_carat) + jnp.sin(c2 * x_carat))
    )


def asymmetry_trafo(
    vector: chex.Array, beta: float, num_dims: int
) -> chex.Array:
    """Assymmetry trafo function for x array input & f value output.
    (p.3; Hansen et al., 2009)"""
    dim = vector.shape[0]

    def get_asy_val(idx, val):
        t = jax.lax.select(num_dims > 1, idx / (num_dims - 1.0), 1.0)
        exp_l0 = 1 + beta * t * (val ** 0.5)
        exp = jax.lax.select(val > 0, exp_l0, 1.0)
        return val ** exp

    return jax.vmap(get_asy_val, in_axes=(0, 0))(jnp.arange(dim), vector)


def boundary_penalty(vector: chex.Array, num_dims: int) -> chex.Array:
    """Penalty for large function deviations to ensure boundary handling.
    (p.3; Hansen et al., 2009)"""
    out = jnp.abs(vector) - 5.0
    mask = jnp.arange(out.shape[0]) < num_dims
    return jnp.sum(jnp.maximum(0.0, out * mask) ** 2)
