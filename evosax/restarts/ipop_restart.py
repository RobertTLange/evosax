import jax
import chex
from functools import partial
from typing import Tuple
from .restarter import RestartWrapper
from .termination import spread_criterion
from .. import Strategies


class IPOP_Restarter(RestartWrapper):
    def __init__(
        self,
        base_strategy,
        stop_criteria=[spread_criterion],
        strategy_kwargs: dict = {},
    ):
        """Increasing-Population Restarts (Auer & Hansen, 2005).
        Reference: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
        """
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
        # Add IPOP-specific state elements to PyTree
        state["active_popsize"] = self.base_strategy.popsize
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
        """Reinstantiate a new strategy with increased population sizes."""
        # Reinstantiate new strategy - based on name of previous strategy
        active_popsize = state["active_popsize"] * params["popsize_multiplier"]

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
        new_state["active_popsize"] = active_popsize
        return new_state
