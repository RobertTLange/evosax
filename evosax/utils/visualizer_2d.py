"""Fitness landscape visualizer and evaluation animator."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# cmap = cm.colors.LinearSegmentedColormap.from_list(
#     "Custom", [(0, "#2f9599"), (0.45, "#eee"), (1, "#8800ff")], N=256
# )

cmap = cm.colors.LinearSegmentedColormap.from_list(
    "Custom", [(0, "#992f2f"), (0.45, "#eee"), (1, "#3a2f99")], N=256
)


class BBOBVisualizer(object):
    """Fitness landscape visualizer and evaluation animator."""

    def __init__(
        self,
        X: chex.Array,
        fitness: chex.Array,
        fn_name: str = "Rastrigin",
        title: str = "",
        use_3d: bool = False,
        plot_log_fn: bool = False,
        seed_id: int = 1,
        interval: int = 50,
        plot_title: bool = True,
        plot_labels: bool = True,
        plot_colorbar: bool = True,
    ):
        from evosax.problems.bbob import BBOB_fns, get_rotation

        self.X = X
        self.fitness = fitness
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

        rng = jax.random.PRNGKey(seed_id)
        rng_q, rng_r = jax.random.split(rng)
        self.R = get_rotation(rng_r, 2)
        self.Q = get_rotation(rng_q, 2)
        self.global_minima = []

        # Set plot configuration
        self.plot_log_fn = plot_log_fn
        self.plot_title = plot_title
        self.plot_labels = plot_labels
        self.plot_colorbar = plot_colorbar

        # Set boundaries for evaluation range of black-box functions
        self.x1_lower_bound, self.x1_upper_bound = -5, 5
        self.x2_lower_bound, self.x2_upper_bound = -5, 5

        # Set meta-data for rotation/azimuth
        self.interval = interval  # Delay between frames in milliseconds.
        try:
            self.num_frames = X.shape[0]
            self.static_frames = int(0.2 * self.num_frames)
            self.azimuths = jnp.linspace(
                0, 89, self.num_frames - self.static_frames
            )
            self.angles = jnp.linspace(
                0, 89, self.num_frames - self.static_frames
            )
        except Exception:
            pass

    def animate(self, save_fname: str):
        """Run animation for provided data."""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_frames,
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        ani.save(save_fname)

    def init(self):
        """Initialize the first frame for the animation."""
        if self.use_3d:
            self.plot_contour_3d()
            (self.scat,) = self.ax.plot(
                self.X[0, :, 0],
                self.X[0, :, 1],
                self.fitness[0, :],
                marker="o",
                c="y",
                linestyle="",
                markersize=10,
                alpha=0.75,
            )

        else:
            self.plot_contour_2d()
            (self.scat,) = self.ax.plot(
                self.X[0, :, 0],
                self.X[0, :, 1],
                marker="o",
                c="y",
                linestyle="",
                markersize=10,
                alpha=0.75,
            )

        return (self.scat,)

    def update(self, frame):
        """Update the frame with the solutions evaluated in generation."""
        # Plot sample points
        self.scat.set_data(self.X[frame, :, 0], self.X[frame, :, 1])
        if self.use_3d:
            if self.plot_log_fn:
                fit = jnp.log(self.fitness[frame, :])
            else:
                fit = self.fitness[frame, :]
            self.scat.set_3d_properties(fit)
            if frame < self.num_frames - self.static_frames:
                self.ax.view_init(self.azimuths[frame], self.angles[frame])

        if self.plot_title:
            if self.plot_log_fn:
                self.ax.set_title(
                    f"Log {self.fn_name}: {self.title} - Generation"
                    f" {frame + 1}",
                    fontsize=15,
                )
            else:
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
        if self.plot_log_fn:
            contour = jnp.log(contour)
        self.ax.contour(X, Y, contour, levels=30, linewidths=0.5, colors="#999")
        im = self.ax.contourf(X, Y, contour, levels=30, cmap=cmap, alpha=0.7)
        if self.plot_title:
            if self.plot_log_fn:
                self.ax.set_title(f"Log {self.fn_name} Function", fontsize=15)
            else:
                self.ax.set_title(f"{self.fn_name} Function", fontsize=15)

        if self.plot_labels:
            self.ax.set_xlabel(r"$x_1$")
            self.ax.set_ylabel(r"$x_2$")
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        if self.plot_colorbar:
            self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()

        if save:
            plt.savefig(f"{self.fn_name}_2d.png", dpi=300)

    def plot_contour_3d(self, save: bool = False):
        """Plot 3d landscape contour."""
        self.fig = plt.figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        x1 = jnp.arange(self.x1_lower_bound, self.x1_upper_bound, 0.01)
        x2 = jnp.arange(self.x2_lower_bound, self.x2_upper_bound, 0.01)
        contour = self.contour_function(x1, x2)
        if self.plot_log_fn:
            contour = jnp.log(contour)

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

        if self.plot_labels:
            self.ax.set_xlabel(r"$x_1$")
            self.ax.set_ylabel(r"$x_2$")
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])

        if self.plot_title:
            if self.plot_log_fn:
                self.ax.set_title(f"Log {self.fn_name} Function", fontsize=15)
                self.ax.set_zlabel(r"$\log f(x)$")
            else:
                self.ax.set_title(f"{self.fn_name} Function", fontsize=15)
                self.ax.set_zlabel(r"$f(x)$")
        self.fig.tight_layout()
        if save:
            plt.savefig(f"{self.fn_name}_3d.png", dpi=300)


if __name__ == "__main__":
    import jax
    from evosax import CMA_ES
    from evosax.problems import BBOBFitness

    # from jax.config import config
    # config.update("jax_enable_x64", True)

    # for fn_name in [
    #     "BuecheRastrigin",
    # ]:  # BBOB_fns.keys():
    #     print(f"Start 2d/3d - {fn_name}")
    #     visualizer = BBOBVisualizer(None, None, fn_name, "")
    #     visualizer.plot_contour_2d(save=True)
    #     visualizer.plot_contour_3d(save=True)

    # Test animations
    # All solutions from single run (10 gens, 16 pmembers, 2 dims)
    for fn_name in [
        "Sphere",
        "RosenbrockOriginal",
        # "RosenbrockRotated",
        "Discus",
        # "RastriginRotated",
        "Schwefel",
        # Large set of functions
        # "BuecheRastrigin",
        # "AttractiveSector",
        # "Weierstrass",
        "SchaffersF7",
        # "GriewankRosenbrock",
        # Part 1: Separable functions
        # "EllipsoidalOriginal",
        # "RastriginOriginal",
        # "LinearSlope",
        # Part 2: Functions with low or moderate conditions
        # "AttractiveSector",
        # "StepEllipsoidal",
        # Part 3: Functions with high conditioning and unimodal
        # "EllipsoidalRotated",
        "BentCigar",
        "SharpRidge",
        "DifferentPowers",
        # Part 4: Multi-modal functions with adequate global structure
        # "SchaffersF7IllConditioned",
        # Part 5: Multi-modal functions with weak global structure
        # "Lunacek",
        # "Gallagher101Me",
        # "Gallagher21Hi",
    ]:
        rng = jax.random.PRNGKey(1)
        strategy = CMA_ES(popsize=4, num_dims=2)
        es_params = strategy.default_params.replace(init_min=-2.5, init_max=2.5)
        es_state = strategy.initialize(rng, es_params)

        problem = BBOBFitness(fn_name, 2)

        X, fitness = [], []
        for g in range(50):
            rng, rng_ask, rng_eval = jax.random.split(rng, 3)
            x, es_state = strategy.ask(rng, es_state, es_params)
            fit = problem.rollout(rng_eval, x)
            es_state = strategy.tell(x, fit, es_state, es_params)
            X.append(x)
            fitness.append(fit)

        X = jnp.stack(X)
        fitness = jnp.stack(fitness)
        print(fn_name, fitness.shape, X.shape)
        visualizer = BBOBVisualizer(
            X,
            fitness,
            fn_name,
            "CMA-ES",
            use_3d=False,
            plot_log_fn=True,
            interval=100,
        )
        visualizer.animate(f"anims/{fn_name}_2d.gif")
        # visualizer = BBOBVisualizer(X, None, "Sphere", "Test Strategy", use_3d=False)
        # visualizer.animate("Sphere_2d.gif")
