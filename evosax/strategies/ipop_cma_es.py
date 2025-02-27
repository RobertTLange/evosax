"""IPOP-CMA-ES (Auer & Hansen, 2005).

Reference: https://ieeexplore.ieee.org/document/1554902
"""

from functools import partial

import jax
from flax import struct

from ..restarts.restarter import (
    WrapperParams,
    WrapperState,
    cma_criterion,
    spread_criterion,
)
from ..types import Fitness, Population, Solution
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
    """IPOP-CMA-ES."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize IPOP-CMA-ES."""
        self.strategy_name = "IPOP_CMA_ES"
        # Instantiate base strategy & wrap it with restart wrapper
        self.strategy = CMA_ES(
            population_size=population_size,
            solution=solution,
            elite_ratio=elite_ratio,
            sigma_init=sigma_init,
            **fitness_kwargs,
        )
        from ..restarts import IPOP_Restarter

        self.wrapped_strategy = IPOP_Restarter(
            self.strategy,
            stop_criteria=[spread_criterion, cma_criterion],
            strategy_kwargs={
                "elite_ratio": elite_ratio,
            },
        )

    @property
    def default_params(self) -> WrapperParams:
        """Return default parameters of evolution strategy."""
        restart_params = self.wrapped_strategy.default_params
        return restart_params.replace(restart_params=RestartParams())

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array, params: WrapperParams | None = None) -> WrapperState:
        return self.wrapped_strategy.init(key, params)

    def ask(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> tuple[jax.Array, WrapperState]:
        x, state = self.wrapped_strategy.ask(key, state, params)
        return x, state

    @partial(jax.jit, static_argnames=("self",))
    def tell(
        self,
        x: Population,
        fitness: Fitness,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> WrapperState:
        return self.wrapped_strategy.tell(x, fitness, state, params)
