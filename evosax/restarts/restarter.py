from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from ..types import ArrayTree, Fitness, Population
from ..utils.eigen_decomp import full_eigen_decomp


@struct.dataclass
class WrapperState:
    strategy_state: ArrayTree
    restart_state: ArrayTree


@struct.dataclass
class WrapperParams:
    strategy_params: ArrayTree
    restart_params: ArrayTree


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
    def init(self, key: jax.Array, params: WrapperParams | None = None) -> WrapperState:
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
    ) -> tuple[jax.Array, WrapperState]:
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
        x: Population,
        fitness: Fitness,
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
        self, fitness: Fitness, state: WrapperState, params: WrapperParams
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


def min_gen_criterion(
    fitness: Fitness, state: WrapperState, params: WrapperParams
) -> bool:
    """Allow stopping if minimal number of generations has passed."""
    min_gen_passed = (
        state.strategy_state.generation_counter >= params.restart_params.min_num_gens
    )
    return min_gen_passed


def spread_criterion(
    fitness: Fitness, state: WrapperState, params: WrapperParams
) -> bool:
    """Stop if min/max fitness spread of recent generation is below thresh."""
    fit_var_too_low = (
        jnp.max(fitness) - jnp.min(fitness) < params.restart_params.min_fitness_spread
    )
    return fit_var_too_low


def cma_criterion(fitness: Fitness, state: WrapperState, params: WrapperParams) -> bool:
    """Termination criterion specific to CMA-ES strategy. Default tolerances:
    tol_x - 1e-12 * sigma
    tol_x_up - 1e4
    tol_condition_C - 1e14
    """
    cma_term = 0
    dC = jnp.diag(state.strategy_state.C)
    # Note: Criterion requires full covariance matrix for decomposition!
    C, B, D = full_eigen_decomp(
        state.strategy_state.C,
        state.strategy_state.B,
        state.strategy_state.D,
    )

    # Stop if std of normal distrib is smaller than tolx in all coordinates
    # and pc is smaller than tolx in all components.
    cond_s_1 = jnp.all(state.strategy_state.sigma * dC < params.restart_params.tol_x)
    cond_s_2 = jnp.all(
        state.strategy_state.sigma * state.strategy_state.p_c
        < params.restart_params.tol_x
    )
    cma_term += jnp.logical_and(cond_s_1, cond_s_2)

    # Stop if detecting divergent behavior -> Stepsize sigma exploded.
    cma_term += state.strategy_state.sigma * jnp.max(D) > params.restart_params.tol_x_up

    # No effect coordinates: stop if adding 0.2-standard deviations
    # in any single coordinate does not change mean.
    cond_no_coord_change = jnp.any(
        state.strategy_state.mean
        == state.strategy_state.mean + (0.2 * state.strategy_state.sigma * jnp.sqrt(dC))
    )
    cma_term += cond_no_coord_change

    # No effect axis: stop if adding 0.1-standard deviation vector in
    # any principal axis direction of C does not change m.
    cond_no_axis_change = jnp.all(
        state.strategy_state.mean
        == state.strategy_state.mean
        + (0.1 * state.strategy_state.sigma * D[0] * B[:, 0])
    )
    cma_term += cond_no_axis_change

    # Stop if the condition number of the covariance matrix exceeds 1e14.
    cond_condition_cov = jnp.max(D) / jnp.min(D) > params.restart_params.tol_condition_C
    cma_term += cond_condition_cov
    return cma_term > 0


def amalgam_criterion(
    fitness: Fitness, state: WrapperState, params: WrapperParams
) -> bool:
    """Termination criterion for iAMaLGaM algorithm (Bosman et al. 2013)"""
    return state.strategy_state.c_mult < 1e-10
