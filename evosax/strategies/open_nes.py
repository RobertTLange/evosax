import jax
import jax.numpy as jnp
from functools import partial
import optax
from .strategy import Strategy


class Open_NES(Strategy):
    def __init__(self, popsize: int, num_dims: int, learning_rate: float):
        super().__init__(num_dims, popsize)
        self.learning_rate = learning_rate
        self.optimizer = optax.chain(
            optax.scale_by_adam(eps=1e-4), optax.scale(-self.learning_rate)
        )

    @property
    def default_params(self):
        """Return default parameters of evolutionary strategy."""
        params = {"sigma_init": 0.1}
        return params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self, rng, params):
        """`initialize` the evolutionary strategy."""
        state = {
            "mean": jnp.zeros(self.num_dims),
            "sigma": params["sigma_init"],
            "gen_counter": 0,
        }
        state["optimizer_state"] = self.optimizer.init(state["mean"])
        return state

    @partial(jax.jit, static_argnums=(0,))
    def ask(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.multivariate_normal(
            rng,
            jnp.zeros(self.num_dims),
            jnp.eye(self.num_dims),
            (int(self.popsize / 2),),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state["mean"] + state["sigma"] * z
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        state["gen_counter"] = state["gen_counter"] + 1

        # Get REINFORCE-style gradient for each sample
        noise = (x - state["mean"]) / state["sigma"]
        nes_grads = 1.0 / (self.popsize * state["sigma"]) * jnp.dot(noise.T, fitness)

        # Natural grad update using optax API!
        updates, opt_state = self.optimizer.update(nes_grads, state["optimizer_state"])
        state["mean"] = optax.apply_updates(state["mean"], updates)
        state["optimizer_state"] = opt_state
        return state


if __name__ == "__main__":
    from evosax.problems import batch_quadratic
    from evosax.utils import FitnessShaper

    rng = jax.random.PRNGKey(0)
    strategy = Open_NES(popsize=50, num_dims=3, learning_rate=0.015)
    fit_shaper = FitnessShaper(z_score_fitness=True)

    params = strategy.default_params
    state = strategy.initialize(rng, params)
    fitness_log = []
    num_iters = 200
    for t in range(num_iters):
        rng, rng_iter = jax.random.split(rng)
        x, state = strategy.ask(rng_iter, state, params)
        fitness = batch_quadratic(x)
        best_id = jnp.argmin(fitness)
        fitness_log.append(fitness[best_id])
        fitness_shaped = fit_shaper.apply(x, fitness)
        state = strategy.tell(x, fitness_shaped, state, params)
