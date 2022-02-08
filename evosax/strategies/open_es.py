import jax
import jax.numpy as jnp
from ..strategy import Strategy
from ..utils import GradientOptimizer


class Open_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int, opt_name: str = "adam"):
        super().__init__(num_dims, popsize)
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)

    @property
    def params_strategy(self):
        """Return default parameters of evolutionary strategy."""
        es_params = {
            "sigma_init": 0.1,
            "sigma_decay": 0.999,
            "sigma_limit": 0.01,
        }
        params = {**es_params, **self.optimizer.default_params}
        return params

    def initialize_strategy(self, rng, params):
        """`initialize` the evolutionary strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.num_dims,),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        es_state = {
            "mean": initialization,
            "sigma": params["sigma_init"],
        }
        state = {**es_state, **self.optimizer.initialize(params)}
        return state

    def ask_strategy(self, rng, state, params):
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (jnp.array(self.popsize / 2, int), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state["mean"] + state["sigma"] * z
        return x, state

    def tell_strategy(self, x, fitness, state, params):
        """`tell` performance data for strategy state update."""
        # Reconstruct noise from last mean/std estimates
        noise = (x - state["mean"]) / state["sigma"]
        theta_grad = (
            1.0 / (self.popsize * state["sigma"]) * jnp.dot(noise.T, fitness)
        )

        # Grad update using optimizer instance - decay lrate if desired
        state = self.optimizer.step(theta_grad, state, params)
        state = self.optimizer.update(state, params)
        state["sigma"] *= params["sigma_decay"]
        state["sigma"] = jnp.maximum(state["sigma"], params["sigma_limit"])
        return state
