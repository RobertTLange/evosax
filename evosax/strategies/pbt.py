import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from ..strategy import Strategy


class PBT(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Synchronous Population-Based Training (Jaderberg et al., 2017)
        Reference: https://arxiv.org/abs/1711.09846"""
        super().__init__(num_dims, popsize)
        self.strategy_name = "PBT"

    @property
    def params_strategy(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        return {
            "noise_scale": 0.1,
            "truncation_selection": 0.2,
            "init_min": 0.0,
            "init_max": 0.1,
        }

    def initialize_strategy(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """
        `initialize` the differential evolution strategy.
        """
        initialization = jax.random.uniform(
            rng,
            (self.popsize, self.num_dims),
            minval=params["init_min"],
            maxval=params["init_max"],
        )
        state = {
            "archive": initialization,
            "fitness": jnp.zeros(self.popsize) - 20e10,
            "copy_id": jnp.zeros(self.popsize, dtype=jnp.int32),
        }
        return state

    def ask_strategy(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """
        `ask` for new proposed candidates to evaluate next.
        Perform explore-exploit step.
        1) Check exploit criterion (e.g. in top 20% of performer).
        2) If not exploit: Copy hyperparams from id and explore/perturb around.
        3) Return new hyperparameters and copy_id (same if exploit)
        """
        rng_members = jax.random.split(rng, self.popsize)
        member_ids = jnp.arange(self.popsize)
        exploit_bool, copy_id, hyperparams = jax.vmap(
            single_member_exploit, in_axes=(0, None, None, None)
        )(member_ids, state["archive"], state["fitness"], params)
        hyperparams = jax.vmap(single_member_explore, in_axes=(0, 0, 0, None))(
            rng_members, exploit_bool, hyperparams, params
        )
        state["copy_id"] = copy_id
        return hyperparams, state

    def tell_strategy(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` update to ES state. - Only copy if improved performance."""
        replace = fitness >= state["fitness"]
        state["archive"] = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state["archive"]
        )
        state["fitness"] = replace * fitness + (1 - replace) * state["fitness"]
        return state


def single_member_exploit(
    member_id: int,
    archive: chex.Array,
    fitness: chex.Array,
    params: chex.ArrayTree,
) -> Tuple[bool, int, chex.Array]:
    """Get the top and bottom performers."""
    best_id = jnp.argmax(fitness)
    exploit_bool = member_id != best_id  # Copy if worker not best
    copy_id = jax.lax.select(exploit_bool, best_id, member_id)
    hyperparams_copy = archive[copy_id]
    return exploit_bool, copy_id, hyperparams_copy


def single_member_explore(
    rng: chex.PRNGKey,
    exploit_bool: int,
    hyperparams: chex.Array,
    params: chex.ArrayTree,
) -> chex.Array:
    """Perform multiplicative noise exploration."""
    explore_noise = (
        jax.random.normal(rng, hyperparams.shape) * params["noise_scale"]
    )
    hyperparams_explore = jax.lax.select(
        exploit_bool, hyperparams + explore_noise, hyperparams
    )
    return hyperparams_explore
