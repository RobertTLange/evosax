import pickle
import jax
import jax.numpy as jnp
import chex
from functools import partial


class ESLog(object):
    def __init__(
        self, num_dims: int, num_generations: int, top_k: int, maximize: bool
    ):
        """Simple jittable logging tool for ES rollouts."""
        self.num_dims = num_dims
        self.num_generations = num_generations
        self.top_k = top_k
        self.maximize = maximize

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self) -> chex.ArrayTree:
        """Initialize the logger storage."""
        log = {
            "top_fitness": jnp.zeros(self.top_k)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_params": jnp.zeros((self.top_k, self.num_dims))
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "gen_counter": 0,
        }
        return log

    # @partial(jax.jit, static_argnums=(0,))
    def update(
        self, log: chex.ArrayTree, x: chex.Array, fitness: chex.Array
    ) -> chex.ArrayTree:
        """Update the logging storage with newest data."""
        # Check if there are solutions better than current archive
        vals = jnp.hstack([log["top_fitness"], fitness])
        params = jnp.vstack([log["top_params"], x])
        top_idx = (
            self.maximize * ((-1) * vals).argsort()
            + ((1 - self.maximize) * vals).argsort()
        )
        log["top_fitness"] = vals[top_idx[: self.top_k]]
        log["top_params"] = params[top_idx[: self.top_k]]
        log["log_top_1"] = (
            log["log_top_1"].at[log["gen_counter"]].set(log["top_fitness"][0])
        )
        log["log_top_mean"] = (
            log["log_top_mean"]
            .at[log["gen_counter"]]
            .set(jnp.mean(log["top_fitness"]))
        )

        log["log_top_std"] = (
            log["log_top_std"]
            .at[log["gen_counter"]]
            .set(jnp.std(log["top_fitness"]))
        )
        log["log_gen_1"] = (
            log["log_gen_1"]
            .at[log["gen_counter"]]
            .set(
                self.maximize * jnp.max(fitness)
                + (1 - self.maximize) * jnp.min(fitness)
            )
        )
        log["log_gen_mean"] = (
            log["log_gen_mean"].at[log["gen_counter"]].set(jnp.mean(fitness))
        )
        log["log_gen_std"] = (
            log["log_gen_std"].at[log["gen_counter"]].set(jnp.std(fitness))
        )
        log["gen_counter"] += 1
        return log

    def save(self, log: chex.ArrayTree, filename: str):
        """Save different parts of logger in .pkl file."""
        with open(filename, "wb") as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str):
        """Reload the pickle logger and return dictionary."""
        with open(filename, "rb") as handle:
            es_logger = pickle.load(handle)
        return es_logger

    def plot(
        self,
        log,
        title,
        ylims=None,
        fig=None,
        ax=None,
        no_legend=False,
    ):
        """Plot fitness trajectory from evo logger over generations."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        int_range = jnp.arange(1, log["gen_counter"] + 1)
        ax.plot(
            int_range, log["log_top_1"][: log["gen_counter"]], label="Top 1"
        )
        ax.plot(
            int_range,
            log["log_top_mean"][: log["gen_counter"]],
            label=f"Top-{self.top_k} Mean",
        )
        ax.plot(
            int_range, log["log_gen_1"][: log["gen_counter"]], label="Gen. 1"
        )
        ax.plot(
            int_range,
            log["log_gen_mean"][: log["gen_counter"]],
            label="Gen. Mean",
        )
        if ylims is not None:
            ax.set_ylim(ylims)
        if not no_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Fitness Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig, ax
