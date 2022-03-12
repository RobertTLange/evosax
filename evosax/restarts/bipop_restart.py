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
    ):
        super().__init__(base_strategy, stop_criteria)
        self.default_popsize = self.base_strategy.popsize

    @property
    def restart_params(self) -> chex.ArrayTree:
        """Return default parameters for strategy restarting."""
        re_params = {
            "min_num_gens": 50,
            "min_fitness_spread": 0.1,
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
        state["restarted"] = False

        # Add BIPOP-specific state elements to PyTree
        state["restart_large_counter"] = 0
        state["active_popsize"] = self.base_strategy.popsize
        state["large_eval_budget"] = 0
        state["small_eval_budget"] = 0
        state["small_pop_active"] = True
        return state

    def ask(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        self.base_strategy = Strategies[self.base_strategy.strategy_name](
            popsize=state["active_popsize"], num_dims=self.num_dims
        )
        x, state = self.base_strategy.ask(rng, state, params)
        return x, state

    def restart_strategy(
        self,
        rng: chex.PRNGKey,
        fitness: chex.Array,
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
        re_state = self.base_strategy.initialize(rng, params)
        re_state["small_pop_active"] = small_pop_active
        re_state["large_eval_budget"] = large_eval_budget
        re_state["small_eval_budget"] = small_eval_budget
        re_state["active_popsize"] = jax.lax.select(
            small_pop_active, small_popsize, large_popsize
        )
        re_state["restart_large_counter"] = (
            state["restart_large_counter"] + 1 - small_pop_active
        )
        return re_state
