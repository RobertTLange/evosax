"""BIPOP-CMA-ES (Hansen, 2009).

[1] https://hal.inria.fr/inria-00382093/document
[2] https://tinyurl.com/44y3ryhf
"""

from collections.abc import Callable
from functools import partial

import jax
from flax import struct

from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

from ...restarts.restarter import (
    WrapperParams,
    WrapperState,
    cma_criterion,
    spread_criterion,
)
from .base import metrics_fn
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


class BIPOP_CMA_ES:
    def __init__(
        self,
        population_size: int,
        solution: Solution,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        # Instantiate base strategy & wrap it with restart wrapper
        self.strategy = CMA_ES(
            population_size=population_size,
            solution=solution,
            elite_ratio=elite_ratio,
            sigma_init=sigma_init,
            fitness_shaping_fn=fitness_shaping_fn,
            metrics_fn=metrics_fn,
        )
        from ..restarts import BIPOP_Restarter

        self.wrapped_strategy = BIPOP_Restarter(
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
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params
        return self.wrapped_strategy.init(key, params)

    def ask(
        self,
        key: jax.Array,
        state: WrapperState,
        params: WrapperParams | None = None,
    ) -> tuple[jax.Array, WrapperState]:
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params
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
        # Use default hyperparameters if no other settings provided
        if params is None:
            params = self.default_params
        return self.wrapped_strategy.tell(x, fitness, state, params)
