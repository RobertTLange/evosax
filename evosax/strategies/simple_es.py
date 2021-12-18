import jax
import jax.numpy as jnp
from ..strategy import Strategy


class Simple_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

    @property
    def params_strategy(self):
        # Only parents have positive weight - equal weighting!
        weights = jnp.zeros(self.popsize)
        weights = jax.ops.index_update(
            weights, jax.ops.index[: self.elite_popsize], 1 / self.elite_popsize
        )
        return {
            "c_m": 1.0,  # Learning rate for population mean
            "c_sigma": 0.1,  # Learning rate for population std
            "weights": weights,  # Weights for population members
            "sigma_init": 1,  # Standard deviation
        }

    def initialize_strategy(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "archive": initialization,
            "fitness": jnp.zeros(self.elite_popsize) - 20e10,
            "mean": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
        }
        return state

    def ask_strategy(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        """
        z = jax.random.normal(rng, (self.popsize, self.num_dims))  # ~ N(0, I)
        x = state["mean"] + state["sigma"] * z  # ~ N(m, σ^2 I)
        return x, state

    def tell_strategy(self, x, fitness, state, params):
        """
        `tell` update to ES state.
        """
        # Sort new results, extract elite, store best performer
        concat_p_f = jnp.hstack([jnp.expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, y_k = update_mean(sorted_solutions, state["mean"], params)
        sigma = update_sigma(y_k, state["sigma"], params)
        state["mean"], state["sigma"] = mean, sigma
        return state


def update_mean(sorted_solutions, mean, params):
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = x_k - mean
    y_w = jnp.sum(y_k.T * params["weights"], axis=1)
    mean_new = mean + params["c_m"] * y_w
    return mean_new, y_k


def update_sigma(y_k, sigma, params):
    """Update stepsize sigma."""
    sigma_est = jnp.sqrt(jnp.sum((y_k.T ** 2 * params["weights"]), axis=1))
    sigma_new = (1 - params["c_sigma"]) * sigma + params["c_sigma"] * sigma_est
    return sigma_new
