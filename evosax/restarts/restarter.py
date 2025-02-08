from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct

from .termination import min_gen_criterion


@struct.dataclass
class WrapperState:
    strategy_state: chex.ArrayTree
    restart_state: chex.ArrayTree


@struct.dataclass
class WrapperParams:
    strategy_params: chex.ArrayTree
    restart_params: chex.ArrayTree


@struct.dataclass
class RestartState:
    restart_counter: int
    restart_next: bool


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50


class RestartWrapper:
    def __init__(self, base_strategy, stop_criteria=[]):
        """Base Class for a Restart Strategy."""
        self.base_strategy = base_strategy
        self.stop_criteria = stop_criteria

    @property
    def num_dims(self) -> int:
        """Get number of problem dimensions from base strategy."""
        return self.base_strategy.num_dims

    @property
    def population_size(self) -> int:
        """Get population size from base strategy."""
        return self.base_strategy.population_size

    @property
    def default_params(self) -> WrapperParams:
        """Return default parameters of evolution strategy."""
        return WrapperParams(
            strategy_params=self.base_strategy.default_params,
            restart_params=self.restart_params,
        )

    @property
    def restart_params(self) -> RestartParams:
        """Return default parameters for strategy restarting."""
        return RestartParams()

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, key: jax.Array, params: WrapperParams | None = None
    ) -> WrapperState:
        """`init` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        strategy_state = self.base_strategy.init(key, params.strategy_params)
        restart_state = RestartState(restart_counter=0, restart_next=False)
        return WrapperState(strategy_state, restart_state)

    def ask(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> tuple[chex.Array, WrapperState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        key_restart, key_ask = jax.random.split(key)
        restart_state = self.restart(key_restart, state, params)
        # Simple tree map - jittable if state dimensions are static
        # Otherwise restart wrapper has to overwrite `ask`
        new_state = jax.tree.map(
            lambda x, y: jax.lax.select(state.restart_state.restart_next, x, y),
            restart_state,
            state,
        )
        # Replace/transfer best member and fitness so far
        strategy_state = new_state.strategy_state.replace(
            best_fitness=state.strategy_state.best_fitness,
            best_member=state.strategy_state.best_member,
        )
        x, strategy_state = self.base_strategy.ask(
            key_ask, strategy_state, params.strategy_params
        )
        return x, new_state.replace(strategy_state=strategy_state)

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> WrapperState:
        """`tell` performance data for strategy state update."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        strategy_state = self.base_strategy.tell(
            x, fitness, state.strategy_state, params.strategy_params
        )
        state = state.replace(strategy_state=strategy_state)
        restart_next = self.stop(fitness, state, params)
        restart_state = state.restart_state.replace(restart_next=restart_next)
        return WrapperState(strategy_state, restart_state)

    @partial(jax.jit, static_argnames=("self",))
    def stop(
        self, fitness: chex.Array, state: WrapperState, params: WrapperParams
    ) -> bool:
        """Check all stopping criteria & return stopping bool indicator."""
        restart_bool = 0
        # Loop over stop criteria functions - stop if one is fullfilled
        for crit in self.stop_criteria:
            restart_bool += crit(fitness, state, params)

        # Check if minimal number of generations has passed!
        return jnp.logical_and(
            restart_bool > 0, min_gen_criterion(fitness, state, params)
        )

    def restart(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams,
    ) -> WrapperState:
        """Restart state for next generations."""
        # Copy over important parts of state from previous strategy
        new_strategy_state = self.restart_strategy(key, state, params)
        new_restart_state = state.restart_state.replace(
            restart_counter=state.restart_state.restart_counter + 1,
            restart_next=False,
        )
        return WrapperState(new_strategy_state, new_restart_state)

    def restart_strategy(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams,
    ) -> WrapperState:
        """Restart strategy specific new state construction."""
        raise NotImplementedError
