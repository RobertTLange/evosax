import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class SimpleES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """Simple Gaussian Evolution Strategy (Rechenberg, 1975)
        Reference: https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.strategy_name = "SimpleES"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        # Only parents have positive weight - equal weighting!
        weights = jnp.zeros(self.popsize)
        weights = weights.at[: self.elite_popsize].set(1 / self.elite_popsize)
        return {
            "c_m": 1.0,  # Learning rate for population mean
            "c_sigma": 0.1,  # Learning rate for population std
            "weights": weights,  # Weights for population members
            "sigma_init": 1.0,  # Standard deviation
            "init_min": 0.0,
            "init_max": 0.0,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.Array
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        state = {
            "mean": jnp.zeros(self.num_dims),
            "sigma": jnp.repeat(params["sigma_init"], self.num_dims),
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new proposed candidates to evaluate next."""
        z = jax.random.normal(rng, (self.popsize, self.num_dims))  # ~ N(0, I)
        x = state["mean"] + state["sigma"] * z  # ~ N(m, σ^2 I)
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` update to ES state."""
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, y_k = update_mean(sorted_solutions, state["mean"], params)
        sigma = update_sigma(y_k, state["sigma"], params)
        state["mean"], state["sigma"] = mean, sigma
        return state


def update_mean(
    sorted_solutions: chex.Array, mean: chex.Array, params: chex.ArrayTree
) -> Tuple[chex.Array, chex.Array]:
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = x_k - mean
    y_w = jnp.sum(y_k.T * params["weights"], axis=1)
    mean_new = mean + params["c_m"] * y_w
    return mean_new, y_k


def update_sigma(
    y_k: chex.Array, sigma: chex.Array, params: chex.ArrayTree
) -> chex.Array:
    """Update stepsize sigma."""
    sigma_est = jnp.sqrt(jnp.sum((y_k.T ** 2 * params["weights"]), axis=1))
    sigma_new = (1 - params["c_sigma"]) * sigma + params["c_sigma"] * sigma_est
    return sigma_new
