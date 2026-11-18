"""Blackbox Optimization Benchmarking Meta-Problem.

[1] https://inria.hal.science/inria-00362633
[2] https://coco-platform.org/testsuites/bbob/overview.html
"""

from functools import partial

import jax
import jax.numpy as jnp
from evosax.types import Fitness, Metrics, Population, Solution
from flax import struct

from ..meta_problem import MetaProblem, Params as MetaParams, State as MetaState
from .bbob_fns import bbob_fns
from .bbob_noise import NoiseModel, NoiseParams


@struct.dataclass
class Params(MetaParams):
    fn_id: jax.Array
    num_dims: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    R: jax.Array
    Q: jax.Array
    noise_params: NoiseParams


@struct.dataclass
class State(MetaState):
    pass


class MetaBBOBProblem(MetaProblem):
    """Blackbox Optimization Benchmarking Meta-Problem class.

    A family of BBOB problem instances. `sample_params` draws an instance: a
    function from `fn_names`, a number of active dimensions, the optimum location
    and value, rotation matrices and noise parameters. Solutions are always flat
    arrays of shape `(max_num_dims,)`; an instance only uses its first
    `params.num_dims` entries.

    Args:
        fn_names: Names of the BBOB functions to sample from, keys of `bbob_fns`
            in `evosax.problems.bbob.bbob_fns`.
        min_num_dims: Lower bound of the number of active dimensions of an
            instance, which is drawn with
            `jax.random.randint(minval=min_num_dims, maxval=max_num_dims)`.
        max_num_dims: Size of the solution vectors and `maxval` of the draw of
            the number of active dimensions.
        noise_config: Keyword arguments of the noise model
            (`evosax.problems.bbob.bbob_noise.NoiseModel`), e.g.
            `"noise_model_names"` (a name or list of names among `"noiseless"`,
            `"gaussian"`, `"uniform"`, `"cauchy"`, `"additive"`) and
            `"use_stabilization"`.

    References:
        - [BBOB function definitions](https://inria.hal.science/inria-00362633)
        - [BBOB test suite overview](https://coco-platform.org/testsuites/bbob/overview.html)
    """

    def __init__(
        self,
        fn_names: list[str] = ["sphere"],
        min_num_dims: int = 2,
        max_num_dims: int = 8,
        noise_config: dict = {
            "noise_model_names": "noiseless",
            "use_stabilization": True,
        },
    ):
        self.fn_names = fn_names
        self.min_num_dims = min_num_dims
        self.max_num_dims = max_num_dims
        self.noise_config = noise_config

        self.x_range = [-5.0, 5.0]
        self.x_opt_range = [-4.0, 4.0]
        self.f_opt_range = [-1000, 1000]

        # Collect active BBOB functions
        fn_ids = []
        fns = []
        for counter, fn_name in enumerate(fn_names):
            assert fn_name in bbob_fns, f"Function {fn_name} not found in bbob_fns"
            fn_ids.append(counter)
            fns.append(jax.vmap(bbob_fns[fn_name], in_axes=(0, None, None, None, None)))
        self.fn_ids = jnp.array(fn_ids)
        self.fns = fns

        # Noise
        self.noise_model = NoiseModel(**noise_config)

    @partial(jax.jit, static_argnames=("self",))
    def sample_params(self, key: jax.Array) -> Params:
        """Sample the parameters of a BBOB problem instance.

        Args:
            key: PRNG key.

        Returns:
            Parameters with the function id (index into `fn_names`), the number
            of active dimensions, the optimum location `x_opt` (uniform in
            `x_opt_range`), the optimal value `f_opt` (Cauchy distributed with
            scale 100, clipped to `[-1000, 1000]`), the rotation matrices `R` and
            `Q` and the noise model parameters.
        """
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
        """Initialize the problem state with an evaluation counter of zero."""
        return State(counter=0)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
        params: Params,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of solutions on a BBOB problem instance.

        Args:
            key: PRNG key, used by the noise model.
            solutions: Population to evaluate, an array of shape
                `(population_size, max_num_dims)`.
            state: Current problem state.
            params: Parameters of the problem instance.

        Returns:
            A tuple `(fitness, state, info)`. `fitness` is the function value to
            minimize: the noisy function value plus the boundary penalty plus
            `params.f_opt`. `state` has its counter incremented. `info` contains
            `"fn_val"` (function value without noise, penalty and `f_opt`),
            `"fn_noise"` (function value after the noise model) and `"fn_pen"`
            (boundary penalty).
        """
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

        return (
            fn_noise + fn_pen + params.f_opt,
            State(counter=state.counter + 1),
            {"fn_val": fn_val, "fn_noise": fn_noise, "fn_pen": fn_pen},
        )

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution uniformly from `x_range`.

        Args:
            key: PRNG key.

        Returns:
            An array of shape `(max_num_dims,)`.
        """
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

        Uses a fixed-size matrix of `max_num_dims` and masks the extra dimensions
        to handle variable `num_dims` while remaining jit-compatible.

        Args:
            key: PRNG key.
            num_dims: Number of active dimensions `n`.

        Returns:
            A matrix of shape `(max_num_dims, max_num_dims)` whose leading
            `(num_dims, num_dims)` block holds the sampled rotation.
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
