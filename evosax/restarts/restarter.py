import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Optional
from functools import partial
from .termination import min_gen_criterion
from flax import struct


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


class RestartWrapper(object):
    def __init__(self, base_strategy, stop_criteria=[]):
        """Base Class for a Restart Strategy."""
        self.base_strategy = base_strategy
        self.stop_criteria = stop_criteria

    @property
    def num_dims(self) -> int:
        """Get number of problem dimensions from base strategy."""
        return self.base_strategy.num_dims

    @property
    def popsize(self) -> int:
        """Get population size from base strategy."""
        return self.base_strategy.popsize

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

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: Optional[WrapperParams] = None
    ) -> WrapperState:
        """`initialize` the evolution strategy."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        strategy_state = self.base_strategy.initialize(
            rng, params.strategy_params
        )
        restart_state = RestartState(restart_counter=0, restart_next=False)
        return WrapperState(strategy_state, restart_state)

    # @partial(jax.jit, static_argnums=(0,))
    def ask(
        self,
        rng: chex.PRNGKey,
        state: WrapperState,
        params: Optional[WrapperParams] = None,
    ) -> Tuple[chex.Array, WrapperState]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params

        rng_ask, rng_restart = jax.random.split(rng)
        restart_state = self.restart(rng_restart, state, params)
        # Simple tree map - jittable if state dimensions are static
        # Otherwise restart wrapper has to overwrite `ask`
        new_state = jax.tree_map(
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
            rng_ask, strategy_state, params.strategy_params
        )
        return x, new_state.replace(strategy_state=strategy_state)

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: WrapperState,
        params: Optional[WrapperParams] = None,
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

    @partial(jax.jit, static_argnums=(0,))
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
        rng: chex.PRNGKey,
        state: WrapperState,
        params: WrapperParams,
    ) -> WrapperState:
        """Restart state for next generations."""
        # Copy over important parts of state from previous strategy
        new_strategy_state = self.restart_strategy(rng, state, params)
        new_restart_state = state.restart_state.replace(
            restart_counter=state.restart_state.restart_counter + 1,
            restart_next=False,
        )
        return WrapperState(new_strategy_state, new_restart_state)

    def restart_strategy(
        self,
        rng: chex.PRNGKey,
        state: WrapperState,
        params: WrapperParams,
    ) -> WrapperState:
        """Restart strategy specific new state construction."""
        raise NotImplementedError
