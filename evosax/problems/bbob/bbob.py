"""Blackbox Optimization Benchmarking Problem.

[1] https://inria.hal.science/inria-00362633
[2] https://coco-platform.org/testsuites/bbob/overview.html
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
from evosax.types import Fitness, Metrics, Population, Solution
from flax import struct

from ..problem import Problem, State as ProblemState
from .bbob_fns import fn_names_short_dict
from .meta_bbob import MetaBBOBProblem


@struct.dataclass
class State(ProblemState):
    pass


class BBOBProblem(Problem):
    """Blackbox Optimization Benchmarking (BBOB).

    A single instance of one BBOB function with a fixed number of dimensions. The
    instance (optimum location, optimal value, rotations and noise parameters) is
    sampled once in the constructor from `seed` and reused by every `eval` call.
    Solutions are flat arrays of shape `(num_dims,)`.

    Args:
        fn_name: Name of the BBOB function, a key of `bbob_fns` in
            `evosax.problems.bbob.bbob_fns`, e.g. `"sphere"` or `"rosenbrock"`.
        num_dims: Number of search dimensions.
        x_opt: Location of the optimum. If None, it is sampled uniformly from
            `x_opt_range`.
        f_opt: Function value at the optimum, added to every fitness value. If
            None, it is sampled.
        sample_rotations: If True, use randomly sampled rotation matrices. If
            False, use identity matrices.
        noise_config: Dictionary with the entries `"noise_model_name"` (one of
            `"noiseless"`, `"gaussian"`, `"uniform"`, `"cauchy"`, `"additive"`)
            and `"use_stabilization"` (if True, noise is stabilized close to the
            optimal value).
        seed: Seed of the PRNG key used to sample the problem instance.

    References:
        - [BBOB function definitions](https://inria.hal.science/inria-00362633)
        - [BBOB test suite overview](https://coco-platform.org/testsuites/bbob/overview.html)
    """

    def __init__(
        self,
        fn_name: str = "sphere",
        num_dims: int = 2,
        x_opt: jax.Array | None = None,
        f_opt: float | None = None,
        sample_rotations: bool = True,
        noise_config: dict = {
            "noise_model_name": "noiseless",
            "use_stabilization": True,
        },
        seed: int = 0,
    ):
        self.fn_name = fn_name
        self._num_dims = num_dims

        key = jax.random.key(seed)
        key_params, key_state = jax.random.split(key)

        # Initialize meta-problem params
        self.meta_problem = MetaBBOBProblem(
            fn_names=[fn_name],
            min_num_dims=self._num_dims,
            max_num_dims=self._num_dims,
            noise_config={
                "noise_model_names": noise_config["noise_model_name"],
                "use_stabilization": noise_config["use_stabilization"],
            },
        )
        self._params = self.meta_problem.sample_params(key_params)

        # Set x_opt and f_opt based on provided values
        if x_opt is not None:
            self._params = self._params.replace(x_opt=x_opt)

        if f_opt is not None:
            self._params = self._params.replace(f_opt=f_opt)

        # Set R and Q based on provided values
        if not sample_rotations:
            self._params = self._params.replace(
                R=jnp.eye(self._num_dims),
                Q=jnp.eye(self._num_dims),
            )

        # Initialize meta-problem state
        self._state = self.meta_problem.init(key_state, self._params)

    @property
    def x_range(self):
        """Range of the search space for solutions."""
        return self.meta_problem.x_range

    @property
    def x_opt_range(self):
        """Range for the optimal solution location."""
        return self.meta_problem.x_opt_range

    @property
    def f_opt_range(self):
        """Range for the optimal function value."""
        return self.meta_problem.f_opt_range

    @property
    def x_opt(self):
        """Optimal solution location."""
        return self._params.x_opt

    @property
    def f_opt(self):
        """Optimal function value."""
        return self._params.f_opt

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array) -> State:
        """Initialize the problem state with an evaluation counter of zero."""
        return self.meta_problem.init(key, self._params)

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self, key: jax.Array, solutions: Population, state: State
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of solutions on the BBOB function.

        Args:
            key: PRNG key, used by the noise model.
            solutions: Population to evaluate, an array of shape
                `(population_size, num_dims)`.
            state: Current problem state.

        Returns:
            A tuple `(fitness, state, info)`. `fitness` is the function value to
            minimize: the noisy function value plus the boundary penalty plus
            `f_opt`. `state` has its counter incremented. `info` contains
            `"fn_val"` (function value without noise, penalty and `f_opt`),
            `"fn_noise"` (function value after the noise model) and `"fn_pen"`
            (boundary penalty).
        """
        fn_val, state, info = self.meta_problem.eval(
            key, solutions, state, self._params
        )
        return fn_val, state, info

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution uniformly from `x_range`.

        Args:
            key: PRNG key.

        Returns:
            An array of shape `(num_dims,)`.
        """
        return jax.random.uniform(
            key,
            shape=(self._num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    def visualize_2d(self, key: jax.Array, *, ax=None, logscale=False):
        """Visualize the optimization problem as a 2D contour plot.

        Requires `num_dims == 2`. The location of the optimum is marked with a
        star for most functions.

        Args:
            key: PRNG key used to evaluate the function on the grid.
            ax (matplotlib.axes.Axes | None): Matplotlib axes to draw on. If None,
                a new figure is created.
            logscale (bool): If True, shift the function values to be at least 1
                and use a logarithmic color scale.

        Returns:
            fig (matplotlib.figure.Figure): The Matplotlib figure containing the
            plot.
        """
        assert self._num_dims == 2

        # Create a meshgrid for visualization
        x = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        y = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        X, Y = jnp.meshgrid(x, y)

        # Convert to JAX arrays and reshape for evaluation
        grid = jnp.reshape(jnp.stack([X, Y], axis=-1), (-1, 2))

        # Evaluate the function at each point
        keys = jax.random.split(key, grid.shape[0])
        values, _, _ = self.eval(keys, grid, self._state)
        Z = values.reshape(X.shape)

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        # Use log scale if requested
        if logscale:
            Z = Z - jnp.min(Z) + 1.0
            contour = ax.contourf(
                X, Y, Z, cmap="viridis_r", levels=50, norm=matplotlib.colors.LogNorm()
            )
        else:
            contour = ax.contourf(X, Y, Z, cmap="viridis_r", levels=50)

        fig.colorbar(contour, ax=ax)

        # Set labels and title
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_title(fn_names_short_dict[self.fn_name])

        # Set the axis limits to match x_range
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.x_range[0], self.x_range[1])

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

        # Plot the optimal solution point
        if self.fn_name not in [
            "rosenbrock_rotated",
            "griewank_rosenbrock",
            "schwefel",
        ]:
            ax.scatter(
                self._params.x_opt[0],
                self._params.x_opt[1],
                color="yellow",
                marker="*",
                s=100,
                edgecolors="black",
                linewidths=0.5,  # Reduce the thickness of the black edges
            )

        # Show the plot only if we created it
        if created_fig:
            plt.tight_layout()
            plt.close()

        return fig

    def visualize_3d(self, key: jax.Array, *, ax=None, logscale=False):
        """Visualize the optimization problem as a 3D surface plot.

        Requires `num_dims == 2`.

        Args:
            key: PRNG key used to evaluate the function on the grid.
            ax (matplotlib.axes.Axes | None): Matplotlib 3D axes to draw on. If
                None, a new figure is created.
            logscale (bool): If True, shift the function values to be at least 1
                and plot their logarithm.

        Returns:
            fig (matplotlib.figure.Figure): The Matplotlib figure containing the
            plot.
        """
        assert self._num_dims == 2

        # Create a meshgrid for visualization
        x = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        y = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        X, Y = jnp.meshgrid(x, y)

        # Convert to JAX arrays and reshape for evaluation
        grid = jnp.reshape(jnp.stack([X, Y], axis=-1), (-1, 2))

        # Evaluate the function at each point
        values, _, _ = self.eval(key, grid, self._state)
        Z = values.reshape(X.shape)

        if logscale:
            Z = Z - jnp.min(Z) + 1.0
            Z = jnp.log(Z)

        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "3d"})
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False

        # Plot the surface with viridis colormap (reversed so small values are yellow)
        ax.plot_surface(X, Y, Z, cmap="viridis_r", antialiased=True, alpha=0.8)

        # Set labels and title
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_title(fn_names_short_dict[self.fn_name])

        # Set the axis limits to match x_range
        ax.set_xlim(self.x_range[0], self.x_range[1])
        ax.set_ylim(self.x_range[0], self.x_range[1])

        # Use scientific notation for z-axis or log scale
        ax.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))

        # Show the plot only if we created it
        if created_fig:
            plt.tight_layout()
            plt.close()

        return fig
