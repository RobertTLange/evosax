import jax
import chex
from functools import partial
from typing import Tuple, Optional
from .restarter import RestartWrapper, WrapperState, WrapperParams
from .termination import spread_criterion
from flax import struct


@struct.dataclass
class RestartState:
    restart_counter: int
    restart_next: bool
    active_popsize: int
    restart_large_counter: int
    large_eval_budget: int
    small_eval_budget: int
    small_pop_active: bool


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    popsize_multiplier: int = 2
    copy_mean: bool = False


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

        from .. import Strategies

        global Strategies

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
        restart_state = RestartState(
            restart_counter=0,
            restart_next=False,
            active_popsize=self.base_strategy.popsize,
            restart_large_counter=0,
            large_eval_budget=0,
            small_eval_budget=0,
            small_pop_active=True,
        )
        return WrapperState(strategy_state, restart_state)

    def ask(
        self,
        rng: chex.PRNGKey,
        state: WrapperState,
        params: Optional[WrapperParams] = None,
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params
        # TODO: Cannot jit! Re-definition of strategy with different popsizes.
        # Is there a clever way to mask active members/popsize?
        # Only compile when base strategy is being updated with new popsize.
        rng_ask, rng_restart = jax.random.split(rng)
        if state.restart_state.restart_next:
            state = self.restart(rng_restart, state, params)
        x, strategy_state = self.base_strategy.ask(
            rng_ask, state.strategy_state, params.strategy_params
        )
        return x, state.replace(strategy_state=strategy_state)

    def restart(
        self,
        rng: chex.PRNGKey,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Reinstantiate a new strategy with interlaced population sizes."""
        # Track number of evals depending on active population
        large_eval_budget = jax.lax.select(
            state.restart_state.small_pop_active,
            state.restart_state.large_eval_budget,
            state.restart_state.large_eval_budget
            + state.restart_state.active_popsize
            * state.strategy_state.gen_counter,
        )
        small_eval_budget = jax.lax.select(
            state.restart_state.small_pop_active,
            state.restart_state.small_eval_budget
            + state.restart_state.active_popsize
            * state.strategy_state.gen_counter,
            state.restart_state.small_eval_budget,
        )
        small_pop_active = small_eval_budget < large_eval_budget

        # Update the population size based on active population size
        pop_mult = params.restart_params.popsize_multiplier ** (
            state.restart_state.restart_large_counter + 1
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

        strategy_state = self.base_strategy.initialize(
            rng, params.strategy_params
        )
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
            active_popsize=active_popsize,
            small_pop_active=small_pop_active,
            large_eval_budget=large_eval_budget,
            small_eval_budget=small_eval_budget,
            restart_large_counter=(
                state.restart_state.restart_large_counter + 1 - small_pop_active
            ),
            restart_counter=state.restart_state.restart_counter + 1,
            restart_next=False,
        )
        return WrapperState(strategy_state, restart_state)
