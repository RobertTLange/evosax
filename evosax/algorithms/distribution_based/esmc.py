"""Evolution Strategy with Meta-loss Clipping (Merchant et al., 2021).

Reference: https://arxiv.org/abs/2107.09661
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ...core import GradientOptimizer, OptParams, OptState
from ...types import Fitness, Population, Solution
from .base import DistributionBasedAlgorithm, Params, State, metrics_fn


@struct.dataclass
class State(State):
    mean: jax.Array
    std: jax.Array
    opt_state: OptState
    z: jax.Array


@struct.dataclass
class Params(Params):
    std_init: float
    std_decay: float
    std_limit: float
    opt_params: OptParams
    std_lrate: float  # Learning rate for std
    std_max_change: float  # Clip adaptive std to 20%


class ESMC(DistributionBasedAlgorithm):
    """Evolution Strategy with Meta-loss Clipping (ESMC)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        opt_name: str = "adam",
        lrate_init: float = 0.05,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize ESMC."""
        assert population_size % 2 == 1, "Population size must be odd"
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)

        # Optimizer
        self.optimizer = GradientOptimizer[opt_name](self.num_dims)
        self.lrate_init = lrate_init
        self.lrate_decay = lrate_decay
        self.lrate_limit = lrate_limit

    @property
    def _default_params(self) -> Params:
        opt_params = self.optimizer.default_params.replace(
            lrate_init=self.lrate_init,
            lrate_decay=self.lrate_decay,
            lrate_limit=self.lrate_limit,
        )
        return Params(
            std_init=1.0,
            std_decay=1.0,
            std_limit=0.0,
            opt_params=opt_params,
            std_lrate=0.2,
            std_max_change=0.2,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=params.std_init * jnp.ones(self.num_dims),
            opt_state=self.optimizer.init(params.opt_params),
            z=jnp.zeros((self.population_size // 2, self.num_dims)),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )
        return state

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        # Antithetic sampling
        z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
        z = jnp.concatenate(
            [jnp.zeros((1, self.num_dims)), z_plus, -z_plus]
        )  # TODO: different from original paper
        x = state.mean + state.std[None, ...] * z
        return x, state.replace(z=z_plus)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        fitness_baseline, fitness = fitness[0], fitness[1:]
        fitness_plus = fitness[: (self.population_size - 1) // 2]
        fitness_minus = fitness[(self.population_size - 1) // 2 :]

        # Compute gradient
        delta = jnp.minimum(fitness_plus, fitness_baseline) - jnp.minimum(
            fitness_minus, fitness_baseline
        )
        grad = jnp.dot(state.z.T, delta) / int((self.population_size - 1) / 2)

        # Grad update using optimizer
        mean, opt_state = self.optimizer.step(  # TODO: different from original paper
            state.mean, grad, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Update state
        std = jnp.clip(state.std * params.std_decay, min=params.std_limit)
        return state.replace(mean=mean, std=std, opt_state=opt_state)
