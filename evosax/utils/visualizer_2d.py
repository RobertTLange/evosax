"""Fitness landscape visualizer and evaluation animator."""
import chex
import jax.numpy as jnp
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from evosax.problems.bbob import BBOB_fns, get_rotation

cmap = cm.colors.LinearSegmentedColormap.from_list(
    "Custom", [(0, "#2f9599"), (0.45, "#eee"), (1, "#8800ff")], N=256
)


class BBOBVisualizer(object):
    """Fitness landscape visualizer and evaluation animator."""

    def __init__(
        self,
        X: chex.Array,
        fn_name: str = "Rastrigin",
        title: str = "",
        use_3d: bool = False,
    ):
        self.X = X
        self.title = title
        self.fn_name = fn_name
        self.use_3d = use_3d
        if not self.use_3d:
            self.fig, self.ax = plt.subplots(figsize=(6, 5))
        else:
            self.fig = plt.figure(figsize=(6, 5))
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        self.fn_name = fn_name
        self.fn = BBOB_fns[self.fn_name]
        self.R = jnp.array(get_rotation(2, 0, b"R"))
        self.Q = jnp.array(get_rotation(2, 0, b"Q"))
        self.global_minima = []

        self.x1_lower_bound, self.x1_upper_bound = -5, 5
        self.x2_lower_bound, self.x2_upper_bound = -5, 5

    def animate(self, save_fname: str):
        """Run animation for provided data."""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.X.shape[0],
            init_func=self.init,
            blit=False,
            interval=10,
        )
        ani.save(save_fname)

    def init(self):
        """Initialize the first frame for the animation."""
        if self.use_3d:
            self.plot_contour_3d()
            (self.scat,) = self.ax.plot(
                self.X[0, :, 0],
                self.X[0, :, 1],
                jnp.ones(X.shape[1]) * 0.1,
                marker="o",
                c="r",
                linestyle="",
                markersize=3,
                alpha=0.5,
            )

        else:
            self.plot_contour_2d()
            (self.scat,) = self.ax.plot(
                self.X[0, :, 0],
                self.X[0, :, 1],
                marker="o",
                c="r",
                linestyle="",
                markersize=3,
                alpha=0.5,
            )

        return (self.scat,)

    def update(self, frame):
        """Update the frame with the solutions evaluated in generation."""
        # Plot sample points
        self.scat.set_data(self.X[frame, :, 0], self.X[frame, :, 1])
        if self.use_3d:
            self.scat.set_3d_properties(jnp.ones(X.shape[1]) * 0.1)
        self.ax.set_title(
            f"{self.fn_name}: {self.title} - Generation {frame + 1}",
            fontsize=15,
        )
        self.fig.tight_layout()
        return (self.scat,)

    def contour_function(self, x1, x2):
        """Evaluate vmapped fitness landscape."""

        def fn_val(x1, x2):
            x = jnp.stack([x1, x2])
            return self.fn(x, self.R, self.Q)

        return jax.vmap(jax.vmap(fn_val, in_axes=(0, None)), in_axes=(None, 0))(
            x1, x2
        )

    def plot_contour_2d(self, save: bool = False):
        """Plot 2d landscape contour."""

        if save:
            self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_xlim(self.x1_lower_bound, self.x1_upper_bound)
        self.ax.set_ylim(self.x2_lower_bound, self.x2_upper_bound)
        self.ax.set_xlim(self.x1_lower_bound, self.x1_upper_bound)
        self.ax.set_ylim(self.x2_lower_bound, self.x2_upper_bound)

        # Plot local minimum value
        for m in self.global_minima:
            self.ax.plot(m[0], m[1], "y*", ms=10)
            self.ax.plot(m[0], m[1], "y*", ms=10)

        x1 = jnp.arange(self.x1_lower_bound, self.x1_upper_bound, 0.01)
        x2 = jnp.arange(self.x2_lower_bound, self.x2_upper_bound, 0.01)
        X, Y = np.meshgrid(x1, x2)
        contour = self.contour_function(x1, x2)
        self.ax.contour(X, Y, contour, levels=30, linewidths=0.5, colors="#999")
        im = self.ax.contourf(X, Y, contour, levels=30, cmap=cmap, alpha=0.7)
        self.ax.set_title(f"{self.fn_name} Function", fontsize=15)
        self.ax.set_xlabel(r"$x_1$")
        self.ax.set_ylabel(r"$x_2$")
        self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()

        if save:
            plt.savefig(f"{self.fn_name}_2d.png", dpi=300)

    def plot_contour_3d(self, save: bool = False):
        """Plot 3d landscape contour."""
        if save:
            self.fig = plt.figure(figsize=(6, 5))
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        x1 = jnp.arange(self.x1_lower_bound, self.x1_upper_bound, 0.01)
        x2 = jnp.arange(self.x2_lower_bound, self.x2_upper_bound, 0.01)
        contour = self.contour_function(x1, x2)
        X, Y = np.meshgrid(x1, x2)
        self.ax.contour(
            X,
            Y,
            contour,
            zdir="z",
            offset=np.min(contour),
            levels=30,
            cmap=cmap,
            alpha=0.5,
        )
        self.ax.plot_surface(
            X,
            Y,
            contour,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.7,
        )

        # Rmove fills and set labels
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        self.ax.xaxis.set_tick_params(labelsize=8)
        self.ax.yaxis.set_tick_params(labelsize=8)
        self.ax.zaxis.set_tick_params(labelsize=8)

        self.ax.set_xlabel(r"$x_1$")
        self.ax.set_ylabel(r"$x_2$")
        self.ax.set_zlabel(r"$f(x)$")
        self.ax.set_title(f"{self.fn_name} Function", fontsize=15)
        self.fig.tight_layout()
        if save:
            plt.savefig(f"{self.fn_name}_3d.png", dpi=300)


if __name__ == "__main__":
    import jax
    from jax.config import config

    config.update("jax_enable_x64", True)

    rng = jax.random.PRNGKey(42)

    for fn_name in [
        "BuecheRastrigin",
    ]:  # BBOB_fns.keys():
        print(f"Start 2d/3d - {fn_name}")
        visualizer = BBOBVisualizer(None, fn_name, "")
        visualizer.plot_contour_2d(save=True)
        visualizer.plot_contour_3d(save=True)

    # # Test animations
    # # All solutions from single run (10 gens, 16 pmembers, 2 dims)
    # X = jax.random.normal(rng, shape=(10, 16, 2))
    # visualizer = BBOBVisualizer(X, "Ackley", "Test Strategy", use_3d=True)
    # visualizer.animate("Ackley_3d.gif")
    # visualizer = BBOBVisualizer(X, "Ackley", "Test Strategy", use_3d=False)
    # visualizer.animate("Ackley_2d.gif")
