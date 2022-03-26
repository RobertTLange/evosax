import jax
import chex
from functools import partial
from typing import Tuple
from .restarter import RestartWrapper
from .termination import spread_criterion
from .. import Strategies


class BIPOP_Restarter(RestartWrapper):
    def __init__(
        self,
        base_strategy,
        stop_criteria=[spread_criterion],
        strategy_kwargs: dict = {},
    ):
        """Bi-Population Restarts (Hansen, 2009) - Interlaced population sizes.
        Reference: https://hal.inria.fr/inria-00382093/document
        Inspired by: https://tinyurl.com/44y3ryhf"""
        super().__init__(base_strategy, stop_criteria)
        self.default_popsize = self.base_strategy.popsize
        self.strategy_kwargs = strategy_kwargs

    @property
    def restart_params(self) -> chex.ArrayTree:
        """Return default parameters for strategy restarting."""
        re_params = {
            "min_num_gens": 50,
            "min_fitness_spread": 1e-12,
            "popsize_multiplier": 2,
        }
        return re_params

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        state = self.base_strategy.initialize(rng, params)
        state["restart_counter"] = 0
        state["restart_next"] = False
        # Add BIPOP-specific state elements to PyTree
        state["active_popsize"] = self.base_strategy.popsize
        state["restart_large_counter"] = 0
        state["large_eval_budget"] = 0
        state["small_eval_budget"] = 0
        state["small_pop_active"] = True
        return state

    def ask(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        # TODO: Cannot jit! Re-definition of strategy with different popsizes.
        # Is there a clever way to mask active members/popsize?
        # Only compile when base strategy is being updated with new popsize.
        rng_ask, rng_restart = jax.random.split(rng)
        if state["restart_next"]:
            state = self.restart(rng_restart, state, params)
        x, state = self.base_strategy.ask(rng_ask, state, params)
        return x, state

    def restart_strategy(
        self,
        rng: chex.PRNGKey,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Reinstantiate a new strategy with interlaced population sizes."""
        # Track number of evals depending on active population
        large_eval_budget = jax.lax.select(
            state["small_pop_active"],
            state["large_eval_budget"],
            state["large_eval_budget"]
            + state["active_popsize"] * state["gen_counter"],
        )
        small_eval_budget = jax.lax.select(
            state["small_pop_active"],
            state["small_eval_budget"]
            + state["active_popsize"] * state["gen_counter"],
            state["small_eval_budget"],
        )
        small_pop_active = small_eval_budget < large_eval_budget

        # Update the population size based on active population size
        pop_mult = params["popsize_multiplier"] ** (
            state["restart_large_counter"] + 1
        )
        small_popsize = jax.lax.floor(
            self.default_popsize * pop_mult ** (jax.random.uniform(rng) ** 2)
        ).astype(int)
        large_popsize = self.default_popsize * pop_mult

        # Reinstantiate new strategy - based on name of previous strategy
        active_popsize = jax.lax.select(
            small_pop_active, small_popsize, large_popsize
        )

        # Reinstantiate new ES with new population size
        self.base_strategy = Strategies[self.base_strategy.strategy_name](
            popsize=int(active_popsize),
            num_dims=self.num_dims,
            **self.strategy_kwargs
        )

        new_state = self.base_strategy.initialize(rng, params)
        # Overwrite new state with old preservables
        for k in [
            "best_fitness",
            "best_member",
            "restart_counter",
        ]:
            new_state[k] = state[k]

        new_state["small_pop_active"] = small_pop_active
        new_state["large_eval_budget"] = large_eval_budget
        new_state["small_eval_budget"] = small_eval_budget
        new_state["restart_large_counter"] = (
            state["restart_large_counter"] + 1 - small_pop_active
        )
        new_state["active_popsize"] = active_popsize
        return new_state
