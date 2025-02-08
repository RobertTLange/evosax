from functools import partial

import chex
import jax
from flax import struct

from ..restarts.restarter import WrapperParams, WrapperState
from .cma_es import CMA_ES


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    population_size_multiplier: int = 2
    tol_x: float = 1e-12
    tol_x_up: float = 1e4
    tol_condition_C: float = 1e14
    copy_mean: bool = True


class IPOP_CMA_ES:
    def __init__(
        self,
        population_size: int,
        pholder_params: chex.ArrayTree | chex.Array | None = None,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        **fitness_kwargs: bool | int | float,
    ):
        """IPOP-CMA-ES (Auer & Hansen, 2005).
        Reference: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
        """
        self.strategy_name = "IPOP_CMA_ES"
        # Instantiate base strategy & wrap it with restart wrapper
        self.strategy = CMA_ES(
            population_size=population_size,
            pholder_params=pholder_params,
            elite_ratio=elite_ratio,
            sigma_init=sigma_init,
            mean_decay=mean_decay,
            **fitness_kwargs,
        )
        from ..restarts import IPOP_Restarter
        from ..restarts.termination import cma_criterion, spread_criterion

        self.wrapped_strategy = IPOP_Restarter(
            self.strategy,
            stop_criteria=[spread_criterion, cma_criterion],
            strategy_kwargs={
                "elite_ratio": elite_ratio,
                "mean_decay": mean_decay,
            },
        )

    @property
    def default_params(self) -> WrapperParams:
        """Return default parameters of evolution strategy."""
        re_params = self.wrapped_strategy.default_params
        return re_params.replace(restart_params=RestartParams())

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, key: jax.Array, params: WrapperParams | None = None
    ) -> WrapperState:
        """`initialize` the evolution strategy."""
        return self.wrapped_strategy.initialize(key, params)

    def ask(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> tuple[chex.Array, WrapperState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.wrapped_strategy.ask(key, state, params)
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> WrapperState:
        """`tell` performance data for strategy state update."""
        return self.wrapped_strategy.tell(x, fitness, state, params)
