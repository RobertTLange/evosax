import jax
import chex
from .restarter import RestartWrapper
from .termination import spread_criterion
from flax import struct


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 0.1
    copy_mean: bool = False


class Simple_Restarter(RestartWrapper):
    def __init__(
        self,
        base_strategy,
        stop_criteria=[spread_criterion],
    ):
        """Simple Restart Strategy - Only reinitialize the state."""
        super().__init__(base_strategy, stop_criteria)

    @property
    def restart_params(self) -> RestartParams:
        """Return default parameters for strategy restarting."""
        return RestartParams()

    def restart_strategy(
        self,
        rng: chex.PRNGKey,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Simple restart by state initialization."""
        new_state = self.base_strategy.initialize(rng, params.strategy_params)
        new_state = new_state.replace(
            mean=jax.lax.select(
                params.restart_params.copy_mean,
                state.strategy_state.mean,
                new_state.mean,
            )
        )
        return new_state
