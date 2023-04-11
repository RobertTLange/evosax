from typing import Tuple, Optional, Union
import jax
import chex
from functools import partial
from flax import struct
from .cma_es import CMA_ES
from ..restarts.restarter import WrapperState, WrapperParams


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    popsize_multiplier: int = 2
    tol_x: float = 1e-12
    tol_x_up: float = 1e4
    tol_condition_C: float = 1e14
    copy_mean: bool = True


class IPOP_CMA_ES(object):
    def __init__(
        self,
        popsize: int,
        num_dims: Optional[int] = None,
        pholder_params: Optional[Union[chex.ArrayTree, chex.Array]] = None,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        mean_decay: float = 0.0,
        n_devices: Optional[int] = None,
        **fitness_kwargs: Union[bool, int, float]
    ):
        """IPOP-CMA-ES (Auer & Hansen, 2005).
        Reference: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
        """
        self.strategy_name = "IPOP_CMA_ES"
        # Instantiate base strategy & wrap it with restart wrapper
        self.strategy = CMA_ES(
            popsize=popsize,
            num_dims=num_dims,
            pholder_params=pholder_params,
            elite_ratio=elite_ratio,
            sigma_init=sigma_init,
            mean_decay=mean_decay,
            n_devices=n_devices,
            **fitness_kwargs
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
        self, rng: chex.PRNGKey, params: Optional[WrapperParams] = None
    ) -> WrapperState:
        """`initialize` the evolution strategy."""
        return self.wrapped_strategy.initialize(rng, params)

    def ask(
        self,
        rng: chex.PRNGKey,
        state: WrapperState,
        params: Optional[WrapperParams] = None,
    ) -> Tuple[chex.Array, WrapperState]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.wrapped_strategy.ask(rng, state, params)
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: WrapperState,
        params: Optional[WrapperParams] = None,
    ) -> WrapperState:
        """`tell` performance data for strategy state update."""
        return self.wrapped_strategy.tell(x, fitness, state, params)
