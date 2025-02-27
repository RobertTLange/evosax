"""Policy Gradients with Parameter-Based Exploration (Sehnke et al., 2010).

Reference: https://link.springer.com/chapter/10.1007/978-3-540-87536-9_40
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from ..core import GradientOptimizer, OptParams, OptState
from ..strategy import Params, State, Strategy, metrics_fn
from ..types import Fitness, Population, Solution


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


class PGPE(Strategy):
    """Policy Gradients with Parameter-Based Exploration (PGPE)."""

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        opt_name: str = "adam",
        lrate_init: float = 0.001,
        lrate_decay: float = 1.0,
        lrate_limit: float = 0.001,
        metrics_fn: Callable = metrics_fn,
        **fitness_kwargs: bool | int | float,
    ):
        """Initialize PGPE."""
        assert population_size % 2 == 0, "Population size must be even."
        super().__init__(population_size, solution, metrics_fn, **fitness_kwargs)
        self.strategy_name = "PGPE"

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
            std_lrate=0.1,
            std_max_change=0.2,
        )

    def _init(self, key: jax.Array, params: Params) -> State:
        state = State(
            mean=jnp.full((self.num_dims,), jnp.nan),
            std=jnp.full((self.num_dims,), params.std_init),
            opt_state=self.optimizer.init(params.opt_params),
            z=jnp.zeros((self.population_size, self.num_dims)),
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
        z = jnp.concatenate([z_plus, -z_plus])

        population = state.mean + state.std * z
        return population, state.replace(z=z)

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        z_plus = state.z[: self.population_size // 2]
        fitness_plus = fitness[: self.population_size // 2]
        fitness_minus = fitness[self.population_size // 2 :]

        # Compute gradient for mean
        z_scaled = state.std * z_plus
        # grad_mean = jnp.mean(0.5 * (fitness_plus - fitness_minus)[:, None] * z_scaled, axis=0)
        grad_mean = (
            jnp.dot(fitness_plus - fitness_minus, z_scaled) / self.population_size
        )  # equivalent to the above

        # Compute gradient for std
        baseline = jnp.mean(fitness)
        # grad_std = jnp.mean((0.5 * (fitness_plus + fitness_minus) - baseline) * (z_scaled**2 - state.std**2) / state.std, axis=0)
        grad_std = (
            jnp.dot(
                fitness_plus + fitness_minus - 2 * baseline,
                (z_scaled**2 - state.std**2) / state.std,
            )
            / self.population_size
        )  # equivalent to the above

        # Grad update using optimizer for mean
        mean, opt_state = self.optimizer.step(
            state.mean, grad_mean, state.opt_state, params.opt_params
        )
        opt_state = self.optimizer.update(opt_state, params.opt_params)

        # Grad update for std
        std_max_change = params.std_max_change * state.std
        std_min = state.std - std_max_change
        std_max = state.std + std_max_change

        std = jnp.clip(
            state.std - params.std_lrate * grad_std,
            min=std_min,
            max=std_max,
        )

        # Update state
        std = jnp.clip(std * params.std_decay, min=params.std_limit)
        return state.replace(mean=mean, std=std, opt_state=opt_state)
