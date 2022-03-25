import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class SimpleGA(Strategy):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """Simple Genetic Algorithm (Such et al., 2017)
        Reference: https://arxiv.org/abs/1712.06567
        Inspired by: https://github.com/hardmaru/estool/blob/master/es.py"""

        super().__init__(num_dims, popsize)
        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)
        self.strategy_name = "SimpleGA"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        return {
            "cross_over_rate": 0.5,  # cross-over probability
            "sigma_init": 0.07,  # initial standard deviation
            "sigma_decay": 0.999,  # anneal standard deviation
            "sigma_limit": 0.01,  # stop annealing if less than this
            "init_min": 0.0,
            "init_max": 0.0,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.Array
    ) -> chex.ArrayTree:
        """`initialize` the differential evolution strategy."""
        initialization = jax.random.uniform(
            rng,
            (self.elite_popsize, self.num_dims),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "mean": initialization.mean(axis=0),
            "archive": initialization,
            "fitness": jnp.zeros(self.elite_popsize) + 20e10,
            "sigma": params["sigma_init"],
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """
        `ask` for new proposed candidates to evaluate next.
        1. For each member of elite:
          - Sample two current elite members (a & b)
          - Cross over all dims of a with corresponding one from b
            if random number > co-rate
          - Additionally add noise on top of all elite parameters
        """
        rng, rng_eps, rng_idx_a, rng_idx_b = jax.random.split(rng, 4)
        rng_mate = jax.random.split(rng, self.popsize)
        epsilon = (
            jax.random.normal(rng_eps, (self.popsize, self.num_dims))
            * state["sigma"]
        )
        elite_ids = jnp.arange(self.elite_popsize)
        idx_a = jax.random.choice(rng_idx_a, elite_ids, (self.popsize,))
        idx_b = jax.random.choice(rng_idx_b, elite_ids, (self.popsize,))
        members_a = state["archive"][idx_a]
        members_b = state["archive"][idx_b]
        y = jax.vmap(single_mate, in_axes=(0, 0, 0, None))(
            rng_mate, members_a, members_b, params["cross_over_rate"]
        )
        y += epsilon
        return jnp.squeeze(y), state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """
        `tell` update to ES state.
        If fitness of y <= fitness of x -> replace in population.
        """
        # Combine current elite and recent generation info
        fitness = jnp.concatenate([fitness, state["fitness"]])
        solution = jnp.concatenate([x, state["archive"]])
        # Select top elite from total archive info
        idx = jnp.argsort(fitness)[0 : self.elite_popsize]
        state["fitness"] = fitness[idx]
        state["archive"] = solution[idx]
        # Update mutation epsilon - multiplicative decay
        state["sigma"] = jax.lax.select(
            state["sigma"] > params["sigma_limit"],
            state["sigma"] * params["sigma_decay"],
            state["sigma"],
        )
        # Keep mean across stored archive around for evaluation protocol
        state["mean"] = state["archive"].mean(axis=0)
        return state


def single_mate(
    rng: chex.PRNGKey, a: chex.Array, b: chex.Array, cross_over_rate: float
) -> chex.Array:
    """Only cross-over dims for x% of all dims."""
    idx = jax.random.uniform(rng, (a.shape[0],)) > cross_over_rate
    cross_over_candidate = a * (1 - idx) + b * idx
    return cross_over_candidate
