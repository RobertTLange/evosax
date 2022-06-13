import jax
import chex
from typing import Tuple
from functools import partial
from .cma_es import CMA_ES
from flax import struct


@struct.dataclass
class RestartParams:
    min_num_gens: int = 50
    min_fitness_spread: float = 1e-12
    popsize_multiplier: int = 2
    tol_x: float = 1e-12
    tol_x_up: float = 1e4
    tol_condition_C: float = 1e14


class IPOP_CMA_ES(object):
    def __init__(self, num_dims: int, popsize: int, elite_ratio: float = 0.5):
        """IPOP-CMA-ES (Auer & Hansen, 2005).
        Reference: http://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
        """
        self.strategy_name = "IPOP_CMA_ES"
        # Instantiate base strategy & wrap it with restart wrapper
        self.strategy = CMA_ES(
            num_dims=num_dims, popsize=popsize, elite_ratio=elite_ratio
        )
        from ..restarts import IPOP_Restarter
        from ..restarts.termination import cma_criterion, spread_criterion

        self.wrapped_strategy = IPOP_Restarter(
            self.strategy,
            stop_criteria=[spread_criterion, cma_criterion],
            strategy_kwargs={"elite_ratio": elite_ratio},
        )

    @property
    def default_params(self) -> chex.ArrayTree:
        """Return default parameters of evolution strategy."""
        re_params = self.wrapped_strategy.default_params
        return re_params.replace(restart_params=RestartParams())

    @partial(jax.jit, static_argnums=(0,))
    def initialize(
        self, rng: chex.PRNGKey, params: chex.ArrayTree
    ) -> chex.ArrayTree:
        """`initialize` the evolution strategy."""
        return self.wrapped_strategy.initialize(rng, params)

    def ask(
        self, rng: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree
    ) -> Tuple[chex.Array, chex.ArrayTree]:
        """`ask` for new parameter candidates to evaluate next."""
        x, state = self.wrapped_strategy.ask(rng, state, params)
        strat_params = self.wrapped_strategy.default_params.strategy_params
        params = params.replace(
            strategy_params=params.strategy_params.replace(
                weights_truncated=strat_params.weights_truncated,
                weights=strat_params.weights,
                mu_eff=strat_params.mu_eff,
                c_1=strat_params.c_1,
                c_mu=strat_params.c_mu,
                c_c=strat_params.c_c,
                c_sigma=strat_params.c_sigma,
                d_sigma=strat_params.d_sigma,
            )
        )
        return x, state

    @partial(jax.jit, static_argnums=(0,))
    def tell(
        self,
        x: chex.Array,
        fitness: chex.Array,
        state: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> chex.ArrayTree:
        """`tell` performance data for strategy state update."""
        return self.wrapped_strategy.tell(x, fitness, state, params)
