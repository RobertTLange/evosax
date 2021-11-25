import jax
import jax.numpy as jnp
from ..strategy import Strategy


class PBT_ES(Strategy):
    def __init__(self, num_dims: int, popsize: int):
        """Synchronous version of Population-Based Training."""
        super().__init__(num_dims, popsize)

    @property
    def default_params(self):
        return {
            "noise_scale": 0.1,
            "truncation_selection": 0.2,
            "init_min": -2,  # Param. init range - min
            "init_max": 2,  # Param. init range - max
            "clip_min": -jnp.finfo(jnp.float32).max,
            "clip_max": jnp.finfo(jnp.float32).max,
        }

    def initialize_strategy(self, rng, params):
        """
        `initialize` the differential evolution strategy.
        """
        state = {
            "archive": jax.random.uniform(
                rng,
                (self.popsize, self.num_dims),
                minval=params["init_min"],
                maxval=params["init_max"],
            ),
            "fitness": jnp.zeros(self.popsize) - 20e10,
        }
        return state

    def ask_strategy(self, rng, state, params):
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
        return copy_id, hyperparams, state

    def tell_strategy(self, x, fitness, state, params):
        """
        `tell` update to ES state. - Only copy if perfomance has improved.
        """
        replace = fitness >= state["fitness"]
        state["archive"] = (
            jnp.expand_dims(replace, 1) * x
            + (1 - jnp.expand_dims(replace, 1)) * state["archive"]
        )
        state["fitness"] = replace * fitness + (1 - replace) * state["fitness"]
        return state


def single_member_exploit(member_id, archive, fitness, params):
    """Get the top and bottom performers."""
    best_id = jnp.argmax(fitness)
    exploit_bool = member_id != best_id  # Copy if worker not best
    copy_id = jax.lax.select(exploit_bool, best_id, member_id)
    hyperparams_copy = archive[copy_id]
    return exploit_bool, copy_id, hyperparams_copy


def single_member_explore(rng, exploit_bool, hyperparams, params):
    explore_noise = jax.random.normal(rng, hyperparams.shape) * params["noise_scale"]
    hyperparams_explore = jax.lax.select(
        exploit_bool, hyperparams + explore_noise, hyperparams
    )
    return hyperparams_explore
