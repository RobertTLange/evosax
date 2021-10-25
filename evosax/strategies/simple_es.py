import jax
import jax.numpy as jnp
from functools import partial
from ..strategy import Strategy


class Simple_ES(Strategy):
    def __init__(self, popsize: int, num_dims: int, elite_ratio: float):
        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

    @property
    def default_params(self):
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
            "init_min": -2,  # Param. init range - min
            "init_max": 2,  # Param. init range - min
        }

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        Initialize all population members by randomly sampling
        positions in search-space (defined in `params`).
        """
        state = {
            "archive": jax.random.uniform(
                rng,
                (self.elite_popsize, self.num_dims),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "fitness": jnp.zeros(self.elite_popsize) - 20e10,
            "gen_counter": 0,
            "mean": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
        }
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """
        `ask` for new proposed candidates to evaluate next.
        """
        z = jax.random.normal(rng, (self.popsize, self.num_dims))  # ~ N(0, I)
        x = state["mean"] + state["sigma"] * z  # ~ N(m, σ^2 I)
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """
        `tell` update to ES state.
        """
        state["gen_counter"] += 1
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


if __name__ == "__main__":
    from evosax.problems import batch_rosenbrock

    rng = jax.random.PRNGKey(0)
    strategy = Simple_ES(popsize=10, num_dims=2, elite_ratio=0.5)
    params = strategy.default_params
    state = strategy.initialize(rng, params)
    fitness_log = []
    num_iters = 10
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        y, state = strategy.ask(rng_iter, state, params)
        fitness = batch_rosenbrock(y, 1, 100)
        state = strategy.tell(y, fitness, state, params)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
        print(t, jnp.min(jnp.array(fitness_log)), state["mean"])
# def check_termination(values, params, state):
#     """ Check whether to terminate simple Gaussian search loop. """
#     # Stop if generation fct values of recent generation is below thresh.
#     if (state["gen_counter"] > params["min_generations"]
#         and jnp.max(values) - jnp.min(values) < params["tol_fun"]):
#         print("TERMINATE ----> Convergence/No progress in objective")
#         return True
#     return False
