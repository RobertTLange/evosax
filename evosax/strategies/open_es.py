import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy
from ..utils import GradientOptimizer


class OpenES(Strategy):
    def __init__(self, num_dims: int, popsize: int, opt_name: str = "adam"):
        """OpenAI-ES (Salimans et al. (2017)
        Reference: https://arxiv.org/pdf/1703.03864.pdf
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""
        super().__init__(num_dims, popsize)
        assert not self.popsize & 1, "Population size must be even"
        assert opt_name in ["sgd", "adam", "rmsprop", "clipup"]
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.strategy_name = "OpenES"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        es_params = {
            "sigma_init": 0.04,
            "sigma_decay": 0.999,
            "sigma_limit": 0.01,
            "init_min": 0.0,
            "init_max": 0.0,
        }
        params = {**es_params, **self.optimizer.default_params}
        return params

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
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

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        # Antithetic sampling of noise
        z_plus = jax.random.normal(
            rng,
            (int(self.popsize / 2), self.num_dims),
        )
        z = jnp.concatenate([z_plus, -1.0 * z_plus])
        x = state["mean"] + state["sigma"] * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
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
