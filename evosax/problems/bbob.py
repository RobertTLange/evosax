"""Blackbox Optimization Benchmarking Problem.

[1] https://inria.hal.science/inria-00362633
[2] https://coco-platform.org/testsuites/bbob/overview.html
"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt

from ..types import Fitness, Population, PyTree, Solution
from .bbob_fn import fn_names_short_dict
from .meta_bbob import MetaBBOBProblem
from .problem import Problem


class BBOBProblem(Problem):
    """Blackbox Optimization Benchmarking (BBOB)."""

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
        """Initialize BBOB problem."""
        self.fn_name = fn_name
        self.num_dims = num_dims

        key = jax.random.key(seed)
        key_params, key_state = jax.random.split(key)

        # Initialize meta-problem params
        noise_config["noise_model_names"] = [noise_config.pop("noise_model_name")]
        self.meta_problem = MetaBBOBProblem(
            fn_names=[fn_name],
            min_num_dims=num_dims,
            max_num_dims=num_dims,
            noise_config=noise_config,
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
                R=jnp.eye(num_dims),
                Q=jnp.eye(num_dims),
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

    @partial(jax.jit, static_argnames=("self",))
    def eval(self, key: jax.Array, solutions: Population) -> tuple[Fitness, PyTree]:
        """Evaluate a batch of solutions."""
        fn_val, _, info = self.meta_problem.eval(
            key, solutions, self._state, self._params
        )
        return fn_val, info

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        return jax.random.uniform(
            key,
            shape=(self.num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    def visualize_2d(self, key: jax.Array, *, ax=None, logscale=False):
        """Visualize optimization problem in 2D."""
        assert self.num_dims == 2

        # Create a meshgrid for visualization
        x = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        y = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        X, Y = jnp.meshgrid(x, y)

        # Convert to JAX arrays and reshape for evaluation
        grid = jnp.reshape(jnp.stack([X, Y], axis=-1), (-1, 2))

        # Evaluate the function at each point
        keys = jax.random.split(key, grid.shape[0])
        values, _ = self.eval(keys, grid)
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
        ax.scatter(
            self._params.x_opt[0],
            self._params.x_opt[1],
            color="red",
            marker="*",
            s=100,
            edgecolors="black",
        )

        # Show the plot only if we created it
        if created_fig:
            plt.tight_layout()
            plt.close()

        return fig

    def visualize_3d(self, key: jax.Array, *, ax=None, logscale=False):
        """Visualize optimization problem in 3D."""
        assert self.num_dims == 2

        # Create a meshgrid for visualization
        x = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        y = jnp.linspace(self.x_range[0], self.x_range[1], 1024)
        X, Y = jnp.meshgrid(x, y)

        # Convert to JAX arrays and reshape for evaluation
        grid = jnp.reshape(jnp.stack([X, Y], axis=-1), (-1, 2))

        # Evaluate the function at each point
        values, _ = self.eval(key, grid)
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
