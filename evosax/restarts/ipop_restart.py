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


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    popsize_multiplier: int = 2
    copy_mean: bool = False


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
        )
        return WrapperState(strategy_state, restart_state)

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
        state: WrapperState,
        params: WrapperParams,
    ) -> WrapperState:
        """Reinstantiate a new strategy with increased population sizes."""
        # Reinstantiate new strategy - based on name of previous strategy
        active_popsize = (
            state.restart_state.active_popsize
            * params.restart_params.popsize_multiplier
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
            restart_counter=state.restart_state.restart_counter + 1,
            restart_next=False,
        )
        return WrapperState(strategy_state, restart_state)
