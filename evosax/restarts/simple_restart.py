import chex
from .restarter import RestartWrapper
from .termination import spread_criterion


class Simple_Restarter(RestartWrapper):
    def __init__(
        self,
        base_strategy,
        stop_criteria=[spread_criterion],
    ):
        """Simple Restart Strategy - Only reinitialize the state."""
        super().__init__(base_strategy, stop_criteria)

    @property
    def restart_params(self) -> chex.ArrayTree:
        """Return default parameters for strategy restarting."""
        re_params = {"min_num_gens": 50, "min_fitness_spread": 0.1}
        return re_params

    def restart_strategy(
        self,
        rng: chex.PRNGKey,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """Simple restart by state initialization."""
        new_state = self.base_strategy.initialize(rng, params)
        return new_state
