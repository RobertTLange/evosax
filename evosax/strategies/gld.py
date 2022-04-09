import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class GLD(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Gradientless Descent (Golovin et al., 2019)
        Reference: https://arxiv.org/pdf/1911.06317.pdf"""
        super().__init__(num_dims, popsize)
        self.strategy_name = "GLD"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        return {
            "init_min": 0.0,
            "init_max": 0.0,
            "radius_max": 0.05,
            "radius_min": 0.001,
            "radius_decay": 5,
        }

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
        state = {
            "mean": initialization,
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new proposed candidates to evaluate next."""
        # Sampling of N(0, 1) noise
        z = jax.random.normal(
            rng,
            (self.popsize, self.num_dims),
        )
        # Exponentially decaying sigma scale
        sigma_scale = params["radius_min"] + jnp.exp2(
            -jnp.arange(self.popsize) / params["radius_decay"]
        ) * (params["radius_max"] - params["radius_min"])
        sigma_scale = sigma_scale.reshape(-1, 1)
        # print(state["best_member"].shape, (sigma_scale * z).shape)
        x = state["best_member"] + sigma_scale * z
        return x, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` update to ES state."""
        # No state update needed - everything happens with best_member update
        state["mean"] = state["best_member"]
        return state
