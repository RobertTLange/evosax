import jax
import jax.numpy as jnp
import chex
from functools import partial


class ClassicFitness(object):
    def __init__(
        self,
        fct_name: str = "rosenbrock",
        num_dims: int = 2,
        num_rollouts: int = 1,
        noise_std: float = 0.0,
    ):
        self.fct_name = fct_name
        self.num_dims = num_dims
        self.num_rollouts = num_rollouts
        # Optional - add Gaussian noise to evaluation fitness
        self.noise_std = noise_std
        assert self.num_dims >= 2

        if self.fct_name == "quadratic":
            self.eval = jax.vmap(quadratic_d_dim, 0)
        elif self.fct_name == "rosenbrock":
            self.eval = jax.vmap(rosenbrock_d_dim, 0)
        elif self.fct_name == "ackley":
            self.eval = jax.vmap(ackley_d_dim, 0)
        elif self.fct_name == "griewank":
            self.eval = jax.vmap(griewank_d_dim, 0)
        elif self.fct_name == "rastrigin":
            self.eval = jax.vmap(rastrigin_d_dim, 0)
        elif self.fct_name == "schwefel":
            self.eval = jax.vmap(schwefel_d_dim, 0)
        elif self.fct_name == "himmelblau":
            assert self.num_dims == 2
            self.eval = jax.vmap(himmelblau_2_dim, 0)
        elif self.fct_name == "six-hump":
            assert self.num_dims == 2
            self.eval = jax.vmap(six_hump_camel_2_dim, 0)
        else:
            raise ValueError("Please provide a valid problem name.")

    @partial(jax.jit, static_argnums=(0,))
    def rollout(
        self, rng_input: chex.PRNGKey, eval_params: chex.Array
    ) -> chex.Array:
        """Batch evaluate the proposal points."""
        fitness = self.eval(eval_params).reshape(eval_params.shape[0], 1)
        noise = self.noise_std * jax.random.normal(
            rng_input, (eval_params.shape[0], self.num_rollouts)
        )
        return (fitness + noise).squeeze()


def himmelblau_2_dim(x: chex.Array) -> chex.Array:
    """
    2-dim. Himmelblau function.
    f(x*)=0 - Minima at [3, 2], [-2.81, 3.13],
                        [-3.78, -3.28], [3.58, -1.85]
    """
    x = x
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def six_hump_camel_2_dim(x: chex.Array) -> chex.Array:
    """
    2-dim. 6-Hump Camel function.
    f(x*)=-1.0316 - Minimum at [0.0898, -0.7126], [-0.0898, 0.7126]
    """
    x = x
    p1 = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
    p2 = x[0] * x[1]
    p3 = (-4 + 4 * x[1] ** 2) * x[1] ** 2
    return p1 + p2 + p3


def quadratic_d_dim(x: chex.Array) -> chex.Array:
    """
    Simple 3-dim. quadratic function.
    f(x*)=0 - Minimum at [0.]ˆd
    """
    return jnp.sum(jnp.square(x))


def rosenbrock_d_dim(x: chex.Array) -> chex.Array:
    """
    D-Dim. Rosenbrock function. x_i ∈ [-32.768, 32.768] or x_i ∈ [-5, 10]
    f(x*)=0 - Minumum at x*=a
    """
    a = 1
    b = 100
    x_i, x_sq, x_p = x[:-1], x[:-1] ** 2, x[1:]
    return jnp.sum((a - x_i) ** 2 + b * (x_p - x_sq) ** 2)


def ackley_d_dim(x: chex.Array) -> chex.Array:
    """
    D-Dim. Ackley function. x_i ∈ [-32.768, 32.768]
    f(x*)=0 - Minimum at x*=[0,...,0]
    """
    a = 20
    b = 0.2
    c = 2 * jnp.pi
    return (
        -a * jnp.exp(-b * jnp.sqrt(jnp.mean(x ** 2)))
        - jnp.exp(jnp.mean(jnp.cos(c * x)))
        + a
        + jnp.exp(1)
    )


def griewank_d_dim(x: chex.Array) -> chex.Array:
    """
    D-Dim. Griewank function. x_i ∈ [-600, 600]
    f(x*)=0 - Minimum at x*=[0,...,0]
    """
    return (
        jnp.sum(x ** 2 / 4000)
        - jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, x.shape[0] + 1))))
        + 1
    )


def rastrigin_d_dim(x: chex.Array) -> chex.Array:
    """
    D-Dim. Rastrigin function. x_i ∈ [-5.12, 5.12]
    f(x*)=0 - Minimum at x*=[0,...,0]
    """
    A = 10
    return A * x.shape[0] + jnp.sum(x ** 2 - A * jnp.cos(2 * jnp.pi * x))


def schwefel_d_dim(x: chex.Array) -> chex.Array:
    """
    D-Dim. Schwefel function. x_i ∈ [-500, 500]
    f(x*)=0 - Minimum at x*=[420.9687,...,420.9687]
    """
    return 418.9829 * x.shape[0] - jnp.sum(
        x * jnp.sin(jnp.sqrt(jnp.absolute(x)))
    )
