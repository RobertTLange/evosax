"""Blackbox Optimization Benchmarking Meta-Problem.

[1] https://inria.hal.science/inria-00362633
[2] https://coco-platform.org/testsuites/bbob/overview.html
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from ...types import Fitness, Population, Solution
from ..meta_problem import MetaProblem, Params, State
from .bbob_fns import bbob_fns
from .bbob_noise import NoiseModel, NoiseParams


@struct.dataclass
class Params(Params):
    fn_id: jax.Array
    num_dims: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    R: jax.Array
    Q: jax.Array
    noise_params: NoiseParams


@struct.dataclass
class State(State):
    counter: int = 0


class MetaBBOBProblem(MetaProblem):
    """Blackbox Optimization Benchmarking Meta-Problem class."""

    def __init__(
        self,
        fn_names: list[str] = ["sphere"],
        min_num_dims: int = 2,
        max_num_dims: int = 8,
        noise_config: dict = {},
    ):
        """Initialize Meta-BBOB problem."""
        self.fn_names = fn_names
        self.min_num_dims = min_num_dims
        self.max_num_dims = max_num_dims
        self.noise_config = noise_config

        self.x_range = [-5.0, 5.0]
        self.x_opt_range = [-4.0, 4.0]
        self.f_opt_range = [-1000, 1000]

        # Collect active BBOB functions
        self.fn_ids, self.fns, counter = [], [], 0
        for fn_name, fn in bbob_fns.items():
            if fn_name in fn_names:
                self.fn_ids.append(counter)
                self.fns.append(jax.vmap(fn, in_axes=(0, None, None, None, None)))
                counter += 1
        self.fn_ids = jnp.array(self.fn_ids)

        # Noise
        self.noise_model = NoiseModel(**noise_config)

    @partial(jax.jit, static_argnames=("self",))
    def sample_params(self, key: jax.Array) -> Params:
        """Sample BBOB parameters."""
        key_fn, key_d, key_x, key_f, key_noise = jax.random.split(key, 5)

        # Sample function id
        fn_id = jax.random.choice(key_fn, self.fn_ids)
        num_dims = jax.random.randint(
            key_d, (), minval=self.min_num_dims, maxval=self.max_num_dims
        )

        # Sample optimal solution location and function value
        x_opt = jax.random.uniform(
            key_x,
            shape=(self.max_num_dims,),
            minval=self.x_opt_range[0],
            maxval=self.x_opt_range[1],
        )
        f_opt = jnp.clip(
            100.0 * jax.random.cauchy(key_f, shape=()), min=-1000, max=1000
        )

        # Sample rotation matrices
        key_R, key_Q = jax.random.split(key)
        R = self.generate_random_rotation(key_R, self.max_num_dims)
        Q = self.generate_random_rotation(key_Q, self.max_num_dims)

        # Sample noise model parameters
        noise_params = self.noise_model.sample_params(key_noise)

        return Params(fn_id, num_dims, x_opt, f_opt, R, Q, noise_params)

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, params: Params) -> State:
        """Initialize state."""
        return State(counter=0)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
        params: Params,
    ) -> tuple[Fitness, State]:
        """Evaluate a batch of solutions."""
        fn_val, fn_pen = jax.lax.switch(
            params.fn_id,
            self.fns,
            solutions,
            params.x_opt,
            params.R,
            params.Q,
            params.num_dims,
        )

        # Apply noise
        fn_noise = self.noise_model.apply(key, fn_val, params.noise_params)

        # Add boundary handling penalty and optimal function value
        fn_val = fn_noise + fn_pen + params.f_opt

        return fn_val, state.replace(counter=state.counter + 1), {}

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        return jax.random.uniform(
            key,
            shape=(self.max_num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    def generate_random_rotation(self, key: jax.Array, num_dims: int) -> jax.Array:
        """Generate a random (n, n) rotation matrix uniformly sampled from SO(n).

        This implementation follows the method described in:
        "How to generate a random unitary matrix" [Maris Ozols 2006]
        http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
        https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m

        Uses a fixed-size matrix of max_num_dims and masks the extra dimensions to handle
        variable num_dims while remaining jit-compatible.
        """
        # Generate fixed-size random normal matrix but mask based on num_dims
        random_matrix = jax.random.normal(key, (self.max_num_dims, self.max_num_dims))
        mask = (jnp.arange(self.max_num_dims)[:, None] < num_dims) & (
            jnp.arange(self.max_num_dims)[None, :] < num_dims
        )
        random_matrix = jnp.where(mask, random_matrix, 0.0)

        # Add identity matrix for masked region to ensure valid QR decomposition
        random_matrix = random_matrix + jnp.where(
            ~mask, jnp.eye(self.max_num_dims), 0.0
        )

        # QR decomposition
        orthogonal_matrix, upper_triangular = jnp.linalg.qr(random_matrix)

        # Extract diagonal and create sign correction matrix
        diagonal = jnp.diag(upper_triangular)
        sign_correction = jnp.diag(diagonal / jnp.abs(diagonal))

        # Apply sign correction
        rotation = orthogonal_matrix @ sign_correction

        # Ensure determinant is 1 by possibly flipping first row
        determinant = jnp.linalg.det(rotation)
        rotation = rotation.at[0].multiply(determinant)

        return rotation
