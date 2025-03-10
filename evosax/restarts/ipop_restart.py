"""Increasing Population Size Restart (Auger & Hansen, 2005).

[1] https://ieeexplore.ieee.org/document/1554902
"""

from functools import partial

import jax
from flax import struct

from evosax.algorithms import algorithms

from .restart_conds import RestartParams, RestartState, spread_cond


@struct.dataclass
class RestartState:
    restart_counter: int
    restart_next: bool
    active_population_size: int


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    population_size_multiplier: int = 2
    copy_mean: bool = False


class IPOP_Restarter:
    """Increasing Population Size Restarter (IPOP Restarter)."""

    def __init__(
        self,
        base_strategy,
        stop_criteria=[spread_cond],
        strategy_kwargs: dict = {},
    ):
        """Initialize the IPOP Restart."""
        super().__init__(base_strategy, stop_criteria)
        self.default_population_size = self.base_strategy.population_size
        self.strategy_kwargs = strategy_kwargs

    @property
    def restart_params(self) -> RestartParams:
        """Return default parameters for strategy restarting."""
        return RestartParams()

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, params: RestartParams | None = None) -> RestartState:
        """`init` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        strategy_state = self.base_strategy.init(key, params.strategy_params)
        restart_state = RestartState(
            restart_counter=0,
            restart_next=False,
            active_population_size=self.base_strategy.population_size,
        )
        return RestartState(strategy_state, restart_state)

    def ask(
        self,
        key: jax.Array,
        state: RestartState,
        params: RestartParams | None = None,
    ) -> tuple[jax.Array, RestartState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        # TODO: Cannot jit! Re-definition of strategy with different population sizes.
        # Is there a clever way to mask active members/population_size?
        # Only compile when base strategy is being updated with new population_size.
        key_restart, key_ask = jax.random.split(key)
        if state.restart_state.restart_next:
            state = self.restart(key_restart, state, params)
        x, strategy_state = self.base_strategy.ask(
            key_ask, state.strategy_state, params.strategy_params
        )
        return x, state.replace(strategy_state=strategy_state)

    def restart(
        self,
        key: jax.Array,
        state: RestartState,
        params: RestartParams,
    ) -> RestartState:
        """Reinstantiate a new strategy with increased population sizes."""
        # Reinstantiate new strategy - based on name of previous strategy
        active_population_size = (
            state.restart_state.active_population_size
            * params.restart_params.population_size_multiplier
        )

        # Reinstantiate new ES with new population size
        self.base_strategy = algorithms[self.base_strategy.strategy_name](
            population_size=int(active_population_size),
            solution=self.base_strategy.solution,
            **self.strategy_kwargs,
        )

        strategy_state = self.base_strategy.init(key, params.strategy_params)
        strategy_state = strategy_state.replace(
            mean=jax.lax.select(
                params.restart_params.copy_mean,
                state.strategy_state.mean,
                strategy_state.mean,
            ),
            best_fitness=state.strategy_state.best_fitness,
            best_member=state.strategy_state.best_member,
        )
        # Overwrite new state with old preservables
        restart_state = state.restart_state.replace(
            active_population_size=active_population_size,
            restart_counter=state.restart_state.restart_counter + 1,
            restart_next=False,
        )
        return RestartState(strategy_state, restart_state)
